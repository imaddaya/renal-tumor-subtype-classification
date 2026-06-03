from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
import numpy as np

import torch
import torch.nn as nn
from torchvision import models, transforms
from mamba_ssm import Mamba
from efficient_kan import KAN

app = FastAPI(title="RCC Inference API - newvenv models")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4
CLASS_NAMES = ["chromophobe", "clearcell", "oncocytoma", "papillary"]

MIN_UPLOAD_PATCHES = 70
MAX_UPLOAD_PATCHES = 500
MODEL_NUM_PATCHES = 70
IMAGE_SIZE = 224

WHITE_THRESHOLD = 220
WHITE_DROP_THRESHOLD = 95.0

BASE_DIR = "/home/hpdeadman/Grad_Project/Models"

MODEL_PATH_CANDIDATES = {
    "ResNet18 + Vision Mamba": [
        os.path.join(BASE_DIR, "ResNet18_VisionMamba", "results", "training", "ResNet18_VisionMamba_model.pth"),
        os.path.join(BASE_DIR, "ResNet18_VisionMamba", "results", "ResNet18_VisionMamba_model.pth"),
    ],
    "ResNet18 + Vision Mamba + KAN": [
        os.path.join(BASE_DIR, "ResNet18_VisionMamba_KAN", "results", "training", "ResNet18_VisionMamba_KAN_model.pth"),
        os.path.join(BASE_DIR, "ResNet18_VisionMamba_KAN", "results", "ResNet18_VisionMamba_KAN_model.pth"),
    ],
}

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


def compute_white_tissue_percent(
    pil_img: Image.Image,
    white_threshold: int = WHITE_THRESHOLD,
) -> tuple[float, float]:
    arr = np.array(pil_img.convert("RGB"))
    white_mask = np.all(arr >= white_threshold, axis=2)
    white_percent = float(white_mask.mean() * 100.0)
    tissue_percent = 100.0 - white_percent
    return round(white_percent, 2), round(tissue_percent, 2)


def get_top_prediction(probabilities: dict[str, float]) -> str:
    return max(probabilities.items(), key=lambda x: x[1])[0]


class ResNet18VisionMambaModel(nn.Module):
    def __init__(self, num_classes=4, d_state=16, d_conv=4, expand=2):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.mamba = Mamba(
            d_model=feat_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        b, n, c, h, w = x.shape
        x = x.view(b * n, c, h, w)
        feats = self.backbone(x)
        feats = feats.view(b, n, -1)
        seq_out = self.mamba(feats)
        slide_feats = seq_out.mean(dim=1)
        return self.classifier(slide_feats)


class ResNet18VisionMambaKANModel(nn.Module):
    def __init__(self, num_classes=4, d_state=16, d_conv=4, expand=2):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.mamba = Mamba(
            d_model=feat_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.kan = KAN([feat_dim, 128, num_classes])

    def forward(self, x):
        b, n, c, h, w = x.shape
        x = x.view(b * n, c, h, w)
        feats = self.backbone(x)
        feats = feats.view(b, n, -1)
        seq_out = self.mamba(feats)
        slide_feats = seq_out.mean(dim=1)
        return self.kan(slide_feats)


MODELS: dict[str, nn.Module] = {}


def build_model(model_name: str) -> nn.Module:
    if model_name == "ResNet18 + Vision Mamba":
        return ResNet18VisionMambaModel(num_classes=NUM_CLASSES)
    if model_name == "ResNet18 + Vision Mamba + KAN":
        return ResNet18VisionMambaKANModel(num_classes=NUM_CLASSES)
    raise ValueError(f"Unsupported model: {model_name}")


def resolve_model_path(model_name: str) -> str:
    candidates = MODEL_PATH_CANDIDATES.get(model_name, [])
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Model file not found for {model_name}. Checked: {candidates}")


def get_model(model_name: str) -> nn.Module:
    if model_name in MODELS:
        return MODELS[model_name]

    model_path = resolve_model_path(model_name)
    model = build_model(model_name).to(DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    MODELS[model_name] = model
    return model


def select_best_patches(
    patch_records: list[dict],
) -> tuple[list[dict], list[dict]]:
    dropped_records = [
        record for record in patch_records
        if record["white_percent"] > WHITE_DROP_THRESHOLD
    ]

    kept_records = [
        record for record in patch_records
        if record["white_percent"] <= WHITE_DROP_THRESHOLD
    ]

    # Deterministic ordering: more tissue first, then less white, then upload order
    kept_records.sort(
        key=lambda r: (-r["tissue_percent"], r["white_percent"], r["upload_index"])
    )

    if len(kept_records) < MODEL_NUM_PATCHES:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Not enough usable patches after filtering.",
                "uploaded_count": len(patch_records),
                "dropped_blank_count": len(dropped_records),
                "usable_count": len(kept_records),
                "required_minimum_usable": MODEL_NUM_PATCHES,
                "white_drop_threshold_percent": WHITE_DROP_THRESHOLD,
            },
        )

    selected_records = kept_records[:MODEL_NUM_PATCHES]
    not_used_records = kept_records[MODEL_NUM_PATCHES:]
    dropped_records.extend(not_used_records)

    return selected_records, dropped_records


def prepare_patch_tensors(selected_records: list[dict]) -> torch.Tensor:
    patch_tensors = []

    for record in selected_records:
        image = record["image"].convert("RGB")
        tensor = transform(image)
        patch_tensors.append(tensor)

    if len(patch_tensors) != MODEL_NUM_PATCHES:
        raise ValueError(
            f"Expected exactly {MODEL_NUM_PATCHES} selected patches, got {len(patch_tensors)}."
        )

    patches = torch.stack(patch_tensors, dim=0)
    return patches.unsqueeze(0).to(DEVICE)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "backend": "newvenv",
        "min_upload_patches": MIN_UPLOAD_PATCHES,
        "max_upload_patches": MAX_UPLOAD_PATCHES,
        "model_num_patches": MODEL_NUM_PATCHES,
    }


@app.get("/models")
def list_models():
    return {"models": list(MODEL_PATH_CANDIDATES.keys())}


@app.post("/predict")
async def predict(
    model_name: str = Form(...),
    true_label: str = Form(...),
    images: list[UploadFile] = File(...),
):
    if model_name not in MODEL_PATH_CANDIDATES:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")

    if true_label not in CLASS_NAMES:
        raise HTTPException(status_code=400, detail=f"Unknown true label: {true_label}")

    if not images:
        raise HTTPException(status_code=400, detail="No images were uploaded.")

    uploaded_count = len(images)

    if uploaded_count < MIN_UPLOAD_PATCHES or uploaded_count > MAX_UPLOAD_PATCHES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Number of uploaded patches must be between "
                f"{MIN_UPLOAD_PATCHES} and {MAX_UPLOAD_PATCHES}. "
                f"Received: {uploaded_count}."
            ),
        )

    patch_records = []
    failed_files = []

    for idx, uploaded in enumerate(images):
        try:
            image_bytes = await uploaded.read()
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            white_percent, tissue_percent = compute_white_tissue_percent(pil_image)

            patch_records.append({
                "upload_index": idx,
                "filename": uploaded.filename or f"patch_{idx:03d}",
                "image": pil_image,
                "white_percent": white_percent,
                "tissue_percent": tissue_percent,
            })
        except Exception:
            failed_files.append(uploaded.filename or f"patch_{idx:03d}")

    if failed_files:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Some uploaded files could not be read as images.",
                "failed_files": failed_files,
            },
        )

    selected_records, dropped_records = select_best_patches(patch_records)

    model = get_model(model_name)
    input_tensor = prepare_patch_tensors(selected_records)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0].detach().cpu().numpy().tolist()

    probabilities = {CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)}
    predicted_label = get_top_prediction(probabilities)

    used_patches = [
        {
            "rank": idx + 1,
            "filename": record["filename"],
            "white_percent": record["white_percent"],
            "tissue_percent": record["tissue_percent"],
        }
        for idx, record in enumerate(selected_records)
    ]

    return {
        "model": model_name,
        "true_label": true_label,
        "predicted_label": predicted_label,
        "correct": predicted_label == true_label,
        "patch_count_uploaded": uploaded_count,
        "patch_count_used": len(selected_records),
        "patch_count_dropped": len(dropped_records),
        "white_drop_threshold_percent": WHITE_DROP_THRESHOLD,
        "probabilities": probabilities,
        "selection_summary": {
            "uploaded_count": uploaded_count,
            "usable_count_after_filtering": len(selected_records),
            "dropped_count": len(dropped_records),
            "selection_rule": "sorted by highest tissue_percent to lowest tissue_percent",
            "target_patch_count": MODEL_NUM_PATCHES,
        },
        "used_patches": used_patches,
        "patch_importance_supported": False,
        "patch_importance": None,
        "top_important_patches": None,
    }