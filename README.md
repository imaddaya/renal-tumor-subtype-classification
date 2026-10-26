# Renal Tumor Subtype Classification

University research project comparing six histopathology image-classification architectures: ResNet18, attention MIL, Macenko normalization, KAN, Vision Mamba, and Vision Mamba + KAN. This is an educational experiment, not a clinically validated diagnostic service. Saved metrics and synthetic smoke tests do not establish clinical accuracy.

## CPU quick start

Use Python 3.12, Node.js 24 and Git. Linux or Ubuntu/WSL 2 is recommended. Standard ResNet/MIL models run on CPU; Vision Mamba requires a separate compatible CUDA environment.

```bash
git clone https://github.com/imaddaya/renal-tumor-subtype-classification.git
cd renal-tumor-subtype-classification
python3.12 -m venv .venv-cpu
source .venv-cpu/bin/activate
python -m pip install -r backend_venv/requirements-inference.txt
python -m uvicorn backend_venv.app:app --host 127.0.0.1 --port 8001
```

Run from the repository root unless stated otherwise. The CPU manifest avoids installing the old mixed CUDA environment dump. The separate training manifests add plotting/data-processing dependencies. In PowerShell activate `.\.venv-cpu\Scripts\Activate.ps1`.

In another terminal:

```bash
cd frontend
npm ci
cp .env.example .env.local
npm run dev
```

Open Vite's printed URL, usually http://localhost:5173. PowerShell: `Copy-Item .env.example .env.local`. Start with ResNet18, ResNet18 + MIL, or ResNet18 + MIL + KAN.

## Checkpoints and reference image

`RCC_MODELS_DIR` defaults to this repository's `Models/`. Checkpoint paths below are relative to that directory:

| Model | Checkpoint |
| --- | --- |
| ResNet18 | `ResNet18/results/training/ResNet18_model.pth` |
| ResNet18 + MIL | `ResNet18_MIL/results/training/ResNet18_MIL_model.pth` |
| ResNet18 + MIL + Macenko | `ResNet18_MIL_Macenko/results/training/ResNet18_MIL_Macenko_model.pth` |
| ResNet18 + MIL + KAN | `ResNet18_MIL_KAN/results/training/ResNet18_MIL_KAN_model.pth` |
| ResNet18 + Vision Mamba | `ResNet18_VisionMamba/results/training/ResNet18_VisionMamba_model.pth` |
| ResNet18 + Vision Mamba + KAN | `ResNet18_VisionMamba_KAN/results/training/ResNet18_VisionMamba_KAN_model.pth` |

The older `results/<checkpoint>` layout is accepted too. `/health` reports checkpoint-file availability; this does not prove inference succeeded. Inference loads the full saved state dict with `weights_only=True` and does not download redundant ImageNet weights. Use trusted checkpoints.

**The original Macenko reference JPEG is no longer available.** Its historical path was `data/train/c/DHMC_0040/p_2688_3808.jpg`; the default is retained for documentation, not as a claim that the file exists. The current slicing script emits numbered PNG patches, so running it does not guarantee recovery of the historical JPEG.

Other models and application startup work without this image. Macenko returns an explicit 503 if its reference is missing/unusable. A patch normalization failure returns 422 instead of silently using unnormalized data. To use a real replacement for a new experiment, configure it explicitly and reevaluate/retrain before comparing against the historical Macenko results:

```bash
export RCC_MACENKO_REFERENCE=/absolute/path/to/reference.jpg
export RCC_MODELS_DIR=/absolute/path/to/Models  # optional
```

PowerShell: `$env:RCC_MACENKO_REFERENCE = "C:\path\reference.jpg"`. Relative paths resolve against the repository rather than the current terminal directory. The root `.env.example` documents these variables; Python does not auto-load it. Reference fitting is lazy and cached after a valid Macenko request. No generated/synthetic image is substituted as the original reference.

## Optional CUDA / Vision Mamba backend

Use Linux/WSL with an NVIDIA GPU, a working driver and a CUDA compiler toolkit compatible with the selected wheel. See [PyTorch's installation matrix](https://pytorch.org/get-started/previous-versions/) and [Mamba's upstream requirements](https://github.com/state-spaces/mamba#installation). GPU execution is not verified by the CPU tests.

```bash
python3.12 -m venv .venv-mamba
source .venv-mamba/bin/activate
python -m pip install torch==2.13.0+cu126 torchvision==0.28.0+cu126 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r backend_newvenv/requirements-inference.txt
python -m pip install mamba-ssm==2.3.1 --no-build-isolation
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m uvicorn backend_newvenv.app:app --host 127.0.0.1 --port 8002
```

`mamba-ssm` may compile a platform-specific extension. A CUDA wheel does not install a full compiler toolkit. If installation fails, compare `nvidia-smi`, `nvcc --version`, Python and torch/CUDA versions with upstream support guidance. Keep a working research environment until the replacement has been tested; the CPU models remain an independent route.

## API configuration and use

`frontend/.env.local` supports `VITE_VENV_API_URL` and `VITE_NEWVENV_API_URL`, defaulting to localhost ports 8001 and 8002. Restart Vite after changes. Backend CORS defaults to localhost/127.0.0.1 port 5173; override `RCC_ALLOWED_ORIGINS` with a comma-separated list for other origins. The inference API does not use credential cookies.

Endpoints: `GET /health`, `GET /models`, and multipart `POST /predict` with `model_name`, `true_label`, and repeated `images` fields. Upload 70–500 patches from one WSI/sample. The service removes very white patches, ranks usable patches deterministically, selects 70 and returns probabilities. MIL attention weights are model outputs, not established clinical explanations.

## Dataset preparation and training

Request dataset access from the [Dartmouth provider](https://bmirds.github.io/KidneyCancer/) under its applicable terms. The provider emails download links after its access form. Data is not downloaded by these scripts.

Use the included `data/wsi_metadata.csv` and image filenames matching `wsi_id`. Put original WSI images in `raw_dataset/`, then run from the root in a training environment:

```bash
python data_prep/slice_and_split_wsis.py
python data_prep/create_patch_csv.py
python Tools/audit_patch_whiteness.py
python data_prep/check_counts.py
python test_loader.py
```

Install `backend_venv/requirements_venv.txt` for standard training tools. For CUDA training, install PyTorch first as above, then `backend_newvenv/requirements_newvenv.txt` with `--no-build-isolation`. Each model directory contains training, multisample testing and visualization scripts. Training replaces that model's checkpoint/reports, so back up results first. Run `python compare_all_models.py` after generating all reports to create comparison tables/plots; it uses repository-relative paths.

Dataset rights, patient-level separation, training reproducibility and saved accuracy have not been independently revalidated. Missing original preprocessing assets limit historical comparison.

## Verify

```bash
# CPU environment; tests do not need a real checkpoint or dataset.
python -m pip install httpx
python -m unittest discover -s tests -v
cd frontend
npm run lint
npm run build
```

Eleven CPU/configuration/API tests cover path overrides, startup without the reference, missing weights, upload counts, normalization failures, synthetic Macenko fitting and request tensor/probability shape with a small test model. They do not reproduce training accuracy or CUDA inference. GitHub Actions runs these tests, Python syntax checks, and frontend lint/build. For real inference comparison, use permitted held-out patches and documented preprocessing; synthetic smoke tests only check the software path.
