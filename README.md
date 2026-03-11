# LEDiff Service

Web service for [LEDiff](https://github.com/Hans1984/LEDiff) (CVPR 2025) — HDR imaging via latent diffusion.

Two modes:
- **Inverse Tone Mapping (ITM)**: LDR image (JPEG/PNG) &rarr; HDR (EXR)
- **HDR Generation**: latent code (.npy) &rarr; HDR (EXR)

Two pretrained models:
- **Highlight**: highlight/overexposed recovery
- **Shadow**: shadow/underexposed recovery

Both models are hot-swappable from the UI. Only one model is loaded in VRAM at a time, with full cleanup on switch.

## Quick Start

```bash
git clone --recurse-submodules <repo-url>
cd lediff-service
```

### Model Weights

LEDiff is based on **Stable Diffusion 1.5** (`runwayml/stable-diffusion-v1-5`) with fine-tuned weights.

Two pretrained models are available on Google Drive:

- **Highlight Hallucination** (recommended for overexposed images):
  https://drive.google.com/file/d/1gd9KNmOQ3RH4yvX_Fp4hu2Jko64nbt2j/view?usp=sharing
- **Shadow Hallucination** (for underexposed images):
  https://drive.google.com/file/d/1tMk0rovHSt93wSeIiQ6fxsxPTCiZ6Dmw/view?usp=sharing

#### Automatic download

```bash
cd service
./download_weights.sh              # Downloads both models
./download_weights.sh highlight    # Highlight only
./download_weights.sh shadow       # Shadow only
```

#### Manual installation

1. Download the archives from Google Drive
2. Extract into `service/weights/highlight/` and `service/weights/shadow/`
3. Verify the structure:

```
weights/
  highlight/
    sd_model/                # Extracted archive contents
      model_index.json
      vae/
        merge_model.pth      # FeatureFusion weights (included in archive)
      unet/
      text_encoder/
      tokenizer/
      scheduler/
    merge_model.pth          # Copy from sd_model/vae/merge_model.pth
  shadow/
    sd_model/                # Same structure
    merge_model.pth
```

The `merge_model.pth` (FeatureFusion weights) is located in the `vae/` subdirectory of each archive.
The `download_weights.sh` script handles everything automatically.

Only one model is required to start — the service auto-detects available models.

### Mac

```bash
# Install dependencies (once)
cd service
pip install -r backend/requirements.txt
pip install -e ../vendor/LEDiff

# Run
./run_mac.sh
# -> http://localhost:8001
```

Auto-detects MPS (Apple Silicon) or CPU.

### Vast.ai

```bash
# On the GPU instance (after SSH)
git clone --recurse-submodules <repo-url>
cd lediff-service/service
# Weights are downloaded automatically if missing
./deploy_vastai.sh
# -> http://0.0.0.0:8001
```

Requirements: PyTorch + CUDA instance, port 8001 open, 10+ GB VRAM recommended.

### Docker

```bash
cd service
# Place weights in weights/ before building
docker compose up -d --build
# -> http://localhost:8001
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Model** | highlight | `highlight` or `shadow` — hot-swappable |
| **Mode** | ITM | `itm` (image &rarr; HDR) or `generation` (.npy &rarr; HDR) |
| **Prompt** | "A photograph with natural lighting" | Text description to guide diffusion |
| **Seed** | 42 | Random seed for reproducibility |
| **Inference Steps** | 50 | Denoising steps (10-100). More = better quality, slower |
| **Guidance Scale** | 7.5 | Text guidance strength (1.0-20.0). Higher = closer to prompt |

## API

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/api/health` | Service status, available models, current model |
| POST | `/api/upload` | Upload image (multipart) or .npy |
| POST | `/api/generate/{job_id}` | Start HDR generation (includes `model_type`) |
| POST | `/api/cancel/{job_id}` | Cancel a job |
| GET | `/api/status/{job_id}` | SSE progress stream |
| GET | `/api/result/{job_id}` | Metadata + HDR analysis |
| GET | `/api/hdr-raw/{job_id}` | Raw float32 data (client-side tone mapping) |
| GET | `/api/download/{job_id}` | Download result as EXR |

## Output

Output is a **float32 RGB EXR** (OpenEXR) file, compatible with any HDR viewer (Nuke, DaVinci Resolve, etc.).

The browser preview includes:
- Real-time tone mapping (ACES, Reinhard, Linear)
- Exposure control (-5 to +5 EV) and gamma
- A/B comparator (LDR vs tonemapped HDR)
- Log-domain HDR histogram

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LEDIFF_WEIGHTS_DIR` | `/app/weights` | Directory containing `highlight/` and `shadow/` subdirs |
| `JOB_TTL_HOURS` | 24 | Job retention time in memory |
| `MAX_MEGAPIXELS` | 50 | Upload resolution limit |

## Architecture

```
service/
  backend/app/
    main.py          # FastAPI + SSE + endpoints
    queue.py         # FIFO queue, serialized GPU worker
    inference.py     # LEDiff wrapper (in-memory, hot-swap models)
    analysis.py      # SDR/HDR analysis
    models.py        # Pydantic schemas
  frontend/static/
    index.html       # UI (vanilla HTML/CSS/JS)
    app.js           # Client-side tone mapping + A/B
    style.css        # Dark theme
```

- Single GPU worker (no GPU concurrency)
- One model in VRAM at a time, automatic switch with full cleanup
- FIFO queue with real-time SSE progress and positions
- No temp files during inference (everything in-memory)
- EXR written only at download time

## Troubleshooting

**"No LEDiff models found"**: No models installed. Run `./download_weights.sh` or manually place weights in `weights/highlight/` and/or `weights/shadow/`.

**MPS errors on Mac**: Some operations may not be supported on MPS. The service uses float32 on MPS (no float16). If errors persist, CPU fallback is automatic on Mac.

**Out of memory**: LEDiff uses ~4-6 GB VRAM per model. Switching between highlight and shadow fully releases the previous model's VRAM. Ensure at least 8 GB VRAM on GPU.

**Port already in use**: The service runs on port 8001 (not 8000) to coexist with singlehdr-service.
