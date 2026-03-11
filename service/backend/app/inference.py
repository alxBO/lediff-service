"""LEDiff inference wrapper for in-memory HDR generation.

Replicates the logic of StableDiffusionITMPipeline and StableDiffusionHDRPipeline
without disk I/O, with progress callbacks, and configurable FeatureFusion path.

Supports hot-switching between highlight and shadow models with full VRAM cleanup.
Supports tiled inference for arbitrary input resolutions (not limited to 512x512).
"""

import gc
import logging
import math
import os
import threading
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import least_squares

import Imath
import OpenEXR

logger = logging.getLogger(__name__)

# Tile size presets and VRAM thresholds (GB)
TILE_SIZES = [512, 768, 1024]
TILE_VRAM_THRESHOLDS = {
    512: 0,     # always safe
    768: 8,     # needs ~7 GB
    1024: 14,   # needs ~12 GB
}
DEFAULT_TILE_OVERLAP = 128  # pixels of overlap between adjacent tiles


# ---------------------------------------------------------------------------
# Device detection & VRAM
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _get_vram_gb(device: torch.device) -> float:
    """Return total VRAM in GB. Returns 0 if unknown."""
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        return props.total_memory / (1024 ** 3)
    if device.type == "mps":
        # Apple Silicon — shared memory, assume conservative 10 GB usable
        return 10.0
    return 0.0


def _auto_tile_size(device: torch.device) -> int:
    """Pick the largest tile size that fits in VRAM."""
    vram = _get_vram_gb(device)
    best = 512
    for size in TILE_SIZES:
        if vram >= TILE_VRAM_THRESHOLDS[size]:
            best = size
    logger.info("Auto tile size: %d (VRAM=%.1f GB)", best, vram)
    return best


# ---------------------------------------------------------------------------
# FeatureFusion (copied from LEDiff to avoid import issues)
# ---------------------------------------------------------------------------

class FeatureFusion(nn.Module):
    def __init__(self, in_channels=4, kernel_size=1, padding=0):
        super().__init__()
        self.weight_conv = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size, padding=padding,
            bias=True, groups=in_channels,
        )
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            self.weight_conv.weight.zero_()
            self.weight_conv.bias.data.fill_(1 / 3)

    def forward(self, feat_low, feat_medium, feat_high):
        w1 = self.weight_conv(feat_low)
        w2 = self.weight_conv(feat_medium)
        w3 = self.weight_conv(feat_high)
        weights = torch.stack([w1, w2, w3], dim=2)
        weights = F.softmax(weights, dim=2)
        merged = (weights[:, :, 0] * feat_low +
                  weights[:, :, 1] * feat_medium +
                  weights[:, :, 2] * feat_high)
        return merged


# ---------------------------------------------------------------------------
# Post-processing helpers (from test_hdr_itm.py)
# ---------------------------------------------------------------------------

def _generate_soft_mask(ldr: np.ndarray, thr: float = 0.05) -> np.ndarray:
    """Soft mask for overexposed regions."""
    msk = np.max(ldr, axis=2)
    msk = np.minimum(1.0, np.maximum(0.0, (msk - 1.0 + thr) / thr))
    return np.expand_dims(msk, axis=2)


def _hdr_luminance_residual(params, ldr_vals, hdr_vals):
    gamma, exp = params
    return (ldr_vals ** gamma) * (2.0 ** exp) - hdr_vals


def _optimize_gamma_exposure(ldr: np.ndarray, hdr: np.ndarray, mask: np.ndarray):
    """Optimize gamma and exposure to match LDR to HDR in non-overexposed regions."""
    mask_3ch = np.broadcast_to(mask, ldr.shape)
    flat_mask = mask_3ch.flatten()
    ldr_flat = ldr.flatten()[flat_mask > 0.5]
    hdr_flat = hdr.flatten()[flat_mask > 0.5]
    if ldr_flat.size == 0:
        return 2.4, 0.0
    if ldr_flat.size > 100000:
        idx = np.random.choice(ldr_flat.size, 100000, replace=False)
        ldr_flat = ldr_flat[idx]
        hdr_flat = hdr_flat[idx]
    result = least_squares(
        _hdr_luminance_residual, [2.4, 0.0],
        bounds=([2.4, -np.inf], [2.6, np.inf]),
        args=(ldr_flat, hdr_flat),
    )
    return result.x[0], result.x[1]


def _blend_with_soft_mask(ldr: np.ndarray, hdr: np.ndarray, gamma: float, exp: float):
    """Blend gamma-corrected LDR with diffusion HDR using soft overexposure mask."""
    mask = _generate_soft_mask(ldr)
    non_over_mask = 1.0 - mask
    ldr_adjusted = (ldr ** gamma) * (2.0 ** exp)
    blended = non_over_mask * ldr_adjusted + mask * hdr
    return np.maximum(blended, 0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# EXR I/O
# ---------------------------------------------------------------------------

def save_exr(filepath: str, img: np.ndarray):
    """Save float32 RGB image as EXR."""
    h, w, _ = img.shape
    header = OpenEXR.Header(w, h)
    float_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    header["channels"] = {"R": float_chan, "G": float_chan, "B": float_chan}
    out = OpenEXR.OutputFile(filepath, header)
    out.writePixels({
        "R": img[:, :, 0].astype(np.float32).tobytes(),
        "G": img[:, :, 1].astype(np.float32).tobytes(),
        "B": img[:, :, 2].astype(np.float32).tobytes(),
    })
    out.close()


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def _decode_image_bytes(img_bytes: bytes) -> np.ndarray:
    """Decode image bytes -> float32 RGB in [0, 1] at original resolution."""
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if img_bgr is None:
        raise ValueError("Cannot decode image")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if img_rgb.dtype == np.uint16:
        return img_rgb.astype(np.float32) / 65535.0
    elif img_rgb.dtype == np.uint8:
        return img_rgb.astype(np.float32) / 255.0
    return img_rgb.astype(np.float32)


def _numpy_to_tensor(img_01: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert float32 [0,1] HxWx3 numpy -> (1,3,H,W) tensor in [-1,1]."""
    img_norm = img_01 * 2.0 - 1.0
    return torch.from_numpy(img_norm).float().permute(2, 0, 1).unsqueeze(0).to(device)


def _preprocess_npy_bytes(npy_bytes: bytes, device: torch.device, vae_scaling: float) -> torch.Tensor:
    """Load .npy from bytes -> latent tensor."""
    import io
    arr = np.load(io.BytesIO(npy_bytes))
    tensor = torch.from_numpy(arr).float().to(device)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    return tensor * vae_scaling


# ---------------------------------------------------------------------------
# Tiling helpers
# ---------------------------------------------------------------------------

def _compute_tile_grid(h: int, w: int, tile_size: int, overlap: int) -> Tuple[int, int, int, int]:
    """Compute tile grid dimensions and required padding.

    Returns (tiles_y, tiles_x, pad_h, pad_w).
    """
    stride = tile_size - overlap
    tiles_x = max(1, math.ceil((w - overlap) / stride)) if w > tile_size else 1
    tiles_y = max(1, math.ceil((h - overlap) / stride)) if h > tile_size else 1
    needed_w = (tiles_x - 1) * stride + tile_size
    needed_h = (tiles_y - 1) * stride + tile_size
    pad_w = max(0, needed_w - w)
    pad_h = max(0, needed_h - h)
    return tiles_y, tiles_x, pad_h, pad_w


def _create_tile_weight(tile_size: int, overlap: int) -> np.ndarray:
    """Create a 2D blending weight for a tile. Feathers edges in overlap regions."""
    weight = np.ones((tile_size, tile_size), dtype=np.float32)
    if overlap <= 0:
        return weight
    ramp = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
    for i in range(overlap):
        weight[i, :] *= ramp[i]
        weight[tile_size - 1 - i, :] *= ramp[i]
        weight[:, i] *= ramp[i]
        weight[:, tile_size - 1 - i] *= ramp[i]
    return weight


# ---------------------------------------------------------------------------
# Main pipeline wrapper
# ---------------------------------------------------------------------------

class LEDiffPipeline:
    """Wraps LEDiff's ITM and HDR Generation pipelines for in-memory inference.

    Supports hot-switching between model types (highlight/shadow) with full
    VRAM cleanup between switches. Only one model is loaded at a time.

    Uses tiled inference to support arbitrary input resolutions.
    """

    def __init__(self, weights_dir: str, available_models: Dict[str, dict]):
        self.device = _get_device()
        self._lock = threading.Lock()
        self._weights_dir = weights_dir
        self._available_models = available_models
        self._current_model_type: Optional[str] = None
        self._auto_tile_size = _auto_tile_size(self.device)
        self._vram_gb = _get_vram_gb(self.device)

        self._pipe = None
        self.vae = None
        self.unet = None
        self.scheduler = None
        self.tokenizer = None
        self.text_encoder = None
        self.vae_scale_factor = None
        self.image_processor = None
        self.fusion = None

        logger.info(
            "LEDiffPipeline initialized (device=%s, vram=%.1fGB, auto_tile=%d, available models: %s)",
            self.device, self._vram_gb, self._auto_tile_size, list(available_models.keys()),
        )

    def _load_model(self, model_type: str):
        if self._current_model_type == model_type and self._pipe is not None:
            return

        if self._current_model_type is not None:
            logger.info("Switching model: %s -> %s", self._current_model_type, model_type)
            self._unload_model()
        else:
            logger.info("Loading model: %s", model_type)

        if model_type not in self._available_models:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(self._available_models.keys())}")

        cfg = self._available_models[model_type]
        from diffusers import StableDiffusionITMPipeline

        self._pipe = StableDiffusionITMPipeline.from_pretrained(
            cfg["model_dir"], torch_dtype=torch.float32
        )
        self._pipe.to(self.device)

        self.vae = self._pipe.vae
        self.unet = self._pipe.unet
        self.scheduler = self._pipe.scheduler
        self.tokenizer = self._pipe.tokenizer
        self.text_encoder = self._pipe.text_encoder
        self.vae_scale_factor = self._pipe.vae_scale_factor
        self.image_processor = self._pipe.image_processor

        self.fusion = FeatureFusion(in_channels=4, kernel_size=1, padding=0).to(self.device)
        state_dict = torch.load(cfg["fusion_weights"], map_location=self.device, weights_only=True)
        self.fusion.load_state_dict(state_dict, strict=False)
        self.fusion.eval()

        self._current_model_type = model_type
        logger.info("Model %s loaded on %s.", model_type, self.device)

    def _unload_model(self):
        logger.info("Unloading model: %s", self._current_model_type)
        for attr in ["fusion", "text_encoder", "unet", "vae", "_pipe"]:
            obj = getattr(self, attr, None)
            if obj is not None:
                del obj
                setattr(self, attr, None)
        self.scheduler = None
        self.tokenizer = None
        self.vae_scale_factor = None
        self.image_processor = None
        self._current_model_type = None
        gc.collect()
        self._clear_device_cache()
        gc.collect()

    def _clear_device_cache(self):
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.empty_cache()

    @property
    def current_model_type(self) -> Optional[str]:
        return self._current_model_type

    @property
    def available_model_types(self) -> list:
        return list(self._available_models.keys())

    @property
    def auto_tile_size(self) -> int:
        return self._auto_tile_size

    @property
    def vram_gb(self) -> float:
        return self._vram_gb

    def close(self):
        with self._lock:
            self._unload_model()

    def clear_device_cache(self):
        self._clear_device_cache()

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def _resolve_tile_size(self, tile_size: int) -> int:
        """Resolve tile_size: 0 = auto, otherwise clamp to valid values."""
        if tile_size <= 0:
            return self._auto_tile_size
        # Clamp to nearest valid size
        valid = sorted(TILE_SIZES)
        for s in reversed(valid):
            if tile_size >= s:
                return s
        return valid[0]

    @torch.no_grad()
    def run(
        self,
        input_data: bytes,
        mode: str,
        model_type: str,
        prompt: str,
        seed: int,
        num_inference_steps: int,
        guidance_scale: float,
        tiling: bool = True,
        tile_size: int = 0,
        progress_cb: Callable = lambda *a: None,
    ) -> np.ndarray:
        with self._lock:
            try:
                self._load_model(model_type)
                resolved_tile_size = self._resolve_tile_size(tile_size)
                if mode == "itm":
                    return self._run_itm(
                        input_data, prompt, seed,
                        num_inference_steps, guidance_scale, progress_cb,
                        tiling=tiling, tile_size=resolved_tile_size,
                    )
                elif mode == "generation":
                    return self._run_generation(
                        input_data, prompt, seed,
                        num_inference_steps, guidance_scale, progress_cb,
                    )
                else:
                    raise ValueError(f"Unknown mode: {mode}")
            finally:
                gc.collect()
                self._clear_device_cache()

    # -------------------------------------------------------------------
    # ITM mode (tiled)
    # -------------------------------------------------------------------

    def _run_itm(self, img_bytes, prompt, seed, steps, guidance_scale, progress_cb, tiling=True, tile_size=512):
        device = self.device

        progress_cb("preprocessing", 0.02, "Decoding image...")
        ldr_full = _decode_image_bytes(img_bytes)
        h, w = ldr_full.shape[:2]

        if not tiling or (h <= tile_size and w <= tile_size):
            # No tiling: resize to tile_size x tile_size, process as single tile, resize back
            return self._run_itm_no_tiling(ldr_full, prompt, seed, steps, guidance_scale, progress_cb, tile_size=tile_size)

        # Compute tile grid
        overlap = DEFAULT_TILE_OVERLAP
        stride = tile_size - overlap

        tiles_y, tiles_x, pad_h, pad_w = _compute_tile_grid(h, w, tile_size, overlap)
        total_tiles = tiles_y * tiles_x

        # Pad image if needed (reflect padding for natural edges)
        if pad_h > 0 or pad_w > 0:
            ldr_padded = np.pad(ldr_full, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        else:
            ldr_padded = ldr_full

        logger.info("Tiled inference: %dx%d image -> %dx%d grid (%d tiles of %dpx), pad=(%d,%d)",
                     w, h, tiles_x, tiles_y, total_tiles, tile_size, pad_w, pad_h)

        # Encode prompt once (shared across all tiles)
        progress_cb("encoding", 0.04, "Encoding prompt...")
        prompt_embeds, negative_prompt_embeds = self._encode_prompt(prompt)
        combined_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # Prepare tile blending weight
        tile_weight = _create_tile_weight(tile_size, overlap)

        # Accumulation buffers
        padded_h, padded_w = ldr_padded.shape[:2]
        hdr_accum = np.zeros((padded_h, padded_w, 3), dtype=np.float64)
        weight_accum = np.zeros((padded_h, padded_w), dtype=np.float64)

        # Process each tile
        for tile_idx in range(total_tiles):
            ty = tile_idx // tiles_x
            tx = tile_idx % tiles_x
            y0 = ty * stride
            x0 = tx * stride

            tile_ldr = ldr_padded[y0:y0 + tile_size, x0:x0 + tile_size]

            tile_frac_start = 0.05 + 0.80 * (tile_idx / total_tiles)
            tile_frac_end = 0.05 + 0.80 * ((tile_idx + 1) / total_tiles)

            def tile_progress(stage, frac_within_tile, msg):
                overall = tile_frac_start + (tile_frac_end - tile_frac_start) * frac_within_tile
                progress_cb(stage, overall, f"[Tile {tile_idx+1}/{total_tiles}] {msg}")

            tile_progress("denoising", 0.0, "Processing tile...")

            hdr_tile = self._process_single_tile(
                tile_ldr, combined_embeds, seed, steps, guidance_scale, tile_progress,
            )

            # Accumulate with blending weight
            hdr_accum[y0:y0 + tile_size, x0:x0 + tile_size] += hdr_tile * tile_weight[:, :, None]
            weight_accum[y0:y0 + tile_size, x0:x0 + tile_size] += tile_weight

            # Free tile tensors between tiles
            del hdr_tile
            gc.collect()

        # Normalize and crop
        hdr_raw = (hdr_accum / np.maximum(weight_accum[:, :, None], 1e-8)).astype(np.float32)
        hdr_raw = hdr_raw[:h, :w]

        # Post-processing on full image
        progress_cb("postprocessing", 0.88, "Optimizing gamma and exposure...")
        gamma, exp = _optimize_gamma_exposure(ldr_full, hdr_raw, 1.0 - _generate_soft_mask(ldr_full))
        hdr_blended = _blend_with_soft_mask(ldr_full, hdr_raw, gamma, exp)

        del hdr_accum, weight_accum, ldr_padded, combined_embeds, prompt_embeds, negative_prompt_embeds

        progress_cb("postprocessing", 0.95, "HDR generation complete")
        return hdr_blended

    def _run_itm_no_tiling(self, ldr_full, prompt, seed, steps, guidance_scale, progress_cb, tile_size=512):
        """Process image without tiling — resizes to tile_size x tile_size, then resizes output back."""
        h_orig, w_orig = ldr_full.shape[:2]

        # Resize to tile_size x tile_size
        ldr_resized = cv2.resize(ldr_full, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
        logger.info("No-tiling inference: %dx%d -> %dx%d", w_orig, h_orig, tile_size, tile_size)

        progress_cb("encoding", 0.04, "Encoding prompt...")
        prompt_embeds, negative_prompt_embeds = self._encode_prompt(prompt)
        combined_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        def single_progress(stage, frac, msg):
            overall = 0.05 + 0.80 * frac
            progress_cb(stage, overall, msg)

        hdr_tile = self._process_single_tile(
            ldr_resized, combined_embeds, seed, steps, guidance_scale, single_progress,
        )

        # Resize HDR output back to original resolution
        hdr_raw = cv2.resize(hdr_tile, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)

        # Post-processing on full image
        progress_cb("postprocessing", 0.88, "Optimizing gamma and exposure...")
        gamma, exp = _optimize_gamma_exposure(ldr_full, hdr_raw, 1.0 - _generate_soft_mask(ldr_full))
        hdr_blended = _blend_with_soft_mask(ldr_full, hdr_raw, gamma, exp)

        del hdr_tile, hdr_raw, combined_embeds, prompt_embeds, negative_prompt_embeds

        progress_cb("postprocessing", 0.95, "HDR generation complete")
        return hdr_blended

    def _process_single_tile(
        self,
        tile_ldr_01: np.ndarray,
        combined_embeds: torch.Tensor,
        seed: int,
        steps: int,
        guidance_scale: float,
        progress_cb: Callable,
    ) -> np.ndarray:
        """Process a single tile through the full LEDiff pipeline.

        Tile can be any size divisible by 8 (VAE requirement).
        Returns float32 HDR tile (after exp()).
        """
        device = self.device

        # Encode tile
        image_tensor = _numpy_to_tensor(tile_ldr_01, device)
        latents_tensor = self.vae.encode(image_tensor).latent_dist.mode() * self.vae.config.scaling_factor

        # Latent spatial dimensions (tile_pixels / 8)
        latent_h, latent_w = latents_tensor.shape[2], latents_tensor.shape[3]

        # Prepare random latents matching tile size
        self.scheduler.set_timesteps(steps, device=device)
        timesteps = self.scheduler.timesteps

        torch.manual_seed(seed)
        latents = self._prepare_latents(combined_embeds.dtype, latent_h, latent_w)
        torch.manual_seed(seed + 22)
        latents_low = self._prepare_latents(combined_embeds.dtype, latent_h, latent_w)

        # Denoising loop 1: medium exposure
        progress_cb("denoising", 0.05, "Medium exposure...")
        for i, t in enumerate(timesteps):
            latent_concat = torch.cat([latents, latents_tensor], dim=1)
            latent_input = torch.cat([latent_concat] * 2)
            latent_input = self.scheduler.scale_model_input(latent_input, t)
            noise_pred = self.unet(latent_input, t, encoder_hidden_states=combined_embeds, return_dict=False)[0]
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if (i + 1) % max(1, steps // 5) == 0:
                progress_cb("denoising", 0.05 + 0.35 * ((i + 1) / len(timesteps)),
                            f"Medium: {i+1}/{steps}")

        latents_medium = latents

        # Denoising loop 2: low exposure
        self.scheduler.set_timesteps(steps, device=device)
        timesteps = self.scheduler.timesteps

        progress_cb("denoising", 0.42, "Low exposure...")
        for i, t in enumerate(timesteps):
            latent_concat = torch.cat([latents_low, latents_medium], dim=1)
            latent_input = torch.cat([latent_concat] * 2)
            latent_input = self.scheduler.scale_model_input(latent_input, t)
            noise_pred = self.unet(latent_input, t, encoder_hidden_states=combined_embeds, return_dict=False)[0]
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)
            latents_low = self.scheduler.step(noise_pred, t, latents_low, return_dict=False)[0]

            if (i + 1) % max(1, steps // 5) == 0:
                progress_cb("denoising", 0.42 + 0.35 * ((i + 1) / len(timesteps)),
                            f"Low: {i+1}/{steps}")

        # Fusion + decode
        progress_cb("fusion", 0.80, "Fusing...")
        latent_merged = self.fusion(latents_low, latents_medium, latents_tensor)

        progress_cb("decoding", 0.90, "Decoding...")
        decoded = self.vae.decode(latent_merged / self.vae.config.scaling_factor, return_dict=False)[0]
        decoded = (decoded + 1.0) / 2.0
        decoded = decoded.clamp(0, 1)
        hdr_tile = decoded.squeeze(0).permute(1, 2, 0).cpu().numpy()
        hdr_tile = np.exp(hdr_tile).astype(np.float32)

        # Cleanup
        del latents, latents_low, latents_medium, latent_merged, decoded, image_tensor, latents_tensor

        progress_cb("decoding", 1.0, "Done")
        return hdr_tile

    # -------------------------------------------------------------------
    # Generation mode (no tiling — input is already a latent)
    # -------------------------------------------------------------------

    def _run_generation(self, npy_bytes, prompt, seed, steps, guidance_scale, progress_cb):
        device = self.device

        progress_cb("preprocessing", 0.02, "Loading latent codes...")
        latents_tensor = _preprocess_npy_bytes(npy_bytes, device, self.vae.config.scaling_factor)

        prompt_embeds, negative_prompt_embeds = self._encode_prompt(prompt)
        combined_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        self.scheduler.set_timesteps(steps, device=device)
        timesteps = self.scheduler.timesteps

        torch.manual_seed(seed)
        latents = self._prepare_latents(combined_embeds.dtype)
        torch.manual_seed(seed + 22)
        latents_low = self._prepare_latents(combined_embeds.dtype)

        # Denoising loop 1: medium exposure
        progress_cb("denoising", 0.05, "Denoising (medium exposure)...")
        for i, t in enumerate(timesteps):
            latent_concat = torch.cat([latents, latents_tensor], dim=1)
            latent_input = torch.cat([latent_concat] * 2)
            latent_input = self.scheduler.scale_model_input(latent_input, t)
            noise_pred = self.unet(latent_input, t, encoder_hidden_states=combined_embeds, return_dict=False)[0]
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if (i + 1) % max(1, steps // 10) == 0:
                frac = 0.05 + 0.30 * ((i + 1) / len(timesteps))
                progress_cb("denoising", frac, f"Medium exposure: step {i+1}/{steps}")

        latents_medium = latents

        self.scheduler.set_timesteps(steps, device=device)
        timesteps = self.scheduler.timesteps

        # Denoising loop 2: low exposure
        progress_cb("denoising", 0.38, "Denoising (low exposure)...")
        for i, t in enumerate(timesteps):
            latent_concat = torch.cat([latents_low, latents_medium], dim=1)
            latent_input = torch.cat([latent_concat] * 2)
            latent_input = self.scheduler.scale_model_input(latent_input, t)
            noise_pred = self.unet(latent_input, t, encoder_hidden_states=combined_embeds, return_dict=False)[0]
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)
            latents_low = self.scheduler.step(noise_pred, t, latents_low, return_dict=False)[0]

            if (i + 1) % max(1, steps // 10) == 0:
                frac = 0.38 + 0.30 * ((i + 1) / len(timesteps))
                progress_cb("denoising", frac, f"Low exposure: step {i+1}/{steps}")

        # Feature fusion
        progress_cb("fusion", 0.72, "Fusing exposure latents...")
        latent_merged = self.fusion(latents_low, latents_medium, latents_tensor)

        # Decode
        progress_cb("decoding", 0.80, "Decoding HDR image...")
        decoded = self.vae.decode(latent_merged / self.vae.config.scaling_factor, return_dict=False)[0]
        decoded = (decoded + 1.0) / 2.0
        decoded = decoded.clamp(0, 1)
        hdr_np = decoded.squeeze(0).permute(1, 2, 0).cpu().numpy()
        hdr_raw = np.exp(hdr_np).astype(np.float32)

        del latents, latents_low, latents_medium, latent_merged, decoded, latents_tensor
        del combined_embeds, prompt_embeds, negative_prompt_embeds

        progress_cb("postprocessing", 0.93, "HDR generation complete")
        return hdr_raw

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _encode_prompt(self, prompt: str):
        device = self.device
        tokens = self.tokenizer(
            prompt, padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        )
        prompt_embeds = self.text_encoder(tokens.input_ids.to(device))[0]
        uncond_tokens = self.tokenizer(
            "", padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        )
        negative_embeds = self.text_encoder(uncond_tokens.input_ids.to(device))[0]
        return prompt_embeds, negative_embeds

    def _prepare_latents(self, dtype, latent_h=64, latent_w=64):
        shape = (1, 4, latent_h, latent_w)
        if self.device.type == "mps":
            latents = torch.randn(shape, dtype=dtype, device="cpu").to(self.device)
        else:
            latents = torch.randn(shape, dtype=dtype, device=self.device)
        latents = latents * self.scheduler.init_noise_sigma
        return latents
