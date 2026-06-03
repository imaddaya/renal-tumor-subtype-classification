from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
import numpy as np

import torch
import torch.nn as nn
from torchvision import models, transforms
import torchstain

app = FastAPI(title="RCC Inference API - venv models")

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
    "ResNet18": [
        os.path.join(BASE_DIR, "ResNet18", "results", "training", "ResNet18_model.pth"),
        os.path.join(BASE_DIR, "ResNet18", "results", "ResNet18_model.pth"),
    ],
    "ResNet18 + MIL": [
        os.path.join(BASE_DIR, "ResNet18_MIL", "results", "training", "ResNet18_MIL_model.pth"),
        os.path.join(BASE_DIR, "ResNet18_MIL", "results", "ResNet18_MIL_model.pth"),
    ],
    "ResNet18 + MIL + Macenko": [
        os.path.join(BASE_DIR, "ResNet18_MIL_Macenko", "results", "training", "ResNet18_MIL_Macenko_model.pth"),
        os.path.join(BASE_DIR, "ResNet18_MIL_Macenko", "results", "ResNet18_MIL_Macenko_model.pth"),
    ],
    "ResNet18 + MIL + KAN": [
        os.path.join(BASE_DIR, "ResNet18_MIL_KAN", "results", "training", "ResNet18_MIL_KAN_model.pth"),
        os.path.join(BASE_DIR, "ResNet18_MIL_KAN", "results", "ResNet18_MIL_KAN_model.pth"),
    ],
}

MIL_MODELS = {
    "ResNet18 + MIL",
    "ResNet18 + MIL + Macenko",
    "ResNet18 + MIL + KAN",
}

TARGET_IMAGE_PATH = "/home/hpdeadman/Grad_Project/data/train/c/DHMC_0040/p_2688_3808.jpg"

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

macenko_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 255)
])

normalizer = torchstain.normalizers.MacenkoNormalizer(backend="torch")
target_img = Image.open(TARGET_IMAGE_PATH).convert("RGB")
target_tensor = macenko_transform(target_img)
normalizer.fit(target_tensor)


def macenko_normalize_pil(pil_img: Image.Image) -> Image.Image:
    src = macenko_transform(pil_img)
    norm = normalizer.normalize(src)[0]

    if isinstance(norm, torch.Tensor):
        arr = norm.detach().cpu().numpy()
    else:
        arr = np.array(norm)

    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


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


class SimpleWSIModel(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c, h, w = x.shape
        x = x.view(b * n, c, h, w)
        feats = self.backbone(x)
        feats = feats.view(b, n, -1)
        slide_feats = feats.mean(dim=1)
        return self.classifier(slide_feats)


class AttentionMILModel(nn.Module):
    def __init__(self, num_classes: int = 4, attn_dim: int = 128):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.attention = nn.Sequential(
            nn.Linear(feat_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        b, n, c, h, w = x.shape
        x = x.view(b * n, c, h, w)
        feats = self.backbone(x)
        feats = feats.view(b, n, -1)
        attn_scores = self.attention(feats)
        attn_weights = torch.softmax(attn_scores, dim=1)
        slide_feats = torch.sum(attn_weights * feats, dim=1)
        logits = self.classifier(slide_feats)

        if return_attention:
            return logits, attn_weights.squeeze(-1)

        return logits


class KANLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        self.poly2 = nn.Linear(in_features, out_features, bias=False)
        self.poly3 = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.poly2(x ** 2) + self.poly3(x ** 3)


class KANClassifier(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, num_classes: int):
        super().__init__()
        self.layer1 = KANLinear(in_features, hidden_features)
        self.norm1 = nn.LayerNorm(hidden_features)
        self.act = nn.GELU()
        self.layer2 = KANLinear(hidden_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.layer2(x)
        return x


class AttentionMILKANModel(nn.Module):
    def __init__(self, num_classes: int = 4, attn_dim: int = 128, kan_hidden: int = 128):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.attention = nn.Sequential(
            nn.Linear(feat_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )
        self.classifier = KANClassifier(feat_dim, kan_hidden, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        b, n, c, h, w = x.shape
        x = x.view(b * n, c, h, w)
        feats = self.backbone(x)
        feats = feats.view(b, n, -1)
        attn_scores = self.attention(feats)
        attn_weights = torch.softmax(attn_scores, dim=1)
        slide_feats = torch.sum(attn_weights * feats, dim=1)
        logits = self.classifier(slide_feats)

        if return_attention:
            return logits, attn_weights.squeeze(-1)

        return logits


MODELS: dict[str, nn.Module] = {}


def build_model(model_name: str) -> nn.Module:
    if model_name == "ResNet18":
        return SimpleWSIModel(num_classes=NUM_CLASSES)
    if model_name == "ResNet18 + MIL":
        return AttentionMILModel(num_classes=NUM_CLASSES)
    if model_name == "ResNet18 + MIL + Macenko":
        return AttentionMILModel(num_classes=NUM_CLASSES)
    if model_name == "ResNet18 + MIL + KAN":
        return AttentionMILKANModel(num_classes=NUM_CLASSES)
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
    model_name: str,
) -> tuple[list[dict], list[dict]]:
    dropped_records = [
        record for record in patch_records
        if record["white_percent"] > WHITE_DROP_THRESHOLD
    ]

    kept_records = [
        record for record in patch_records
        if record["white_percent"] <= WHITE_DROP_THRESHOLD
    ]

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


def prepare_patch_tensors(
    selected_records: list[dict],
    model_name: str,
) -> torch.Tensor:
    patch_tensors = []

    for record in selected_records:
        image = record["image"].convert("RGB")

        if model_name == "ResNet18 + MIL + Macenko":
            try:
                image = macenko_normalize_pil(image)
            except Exception:
                pass

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
        "backend": "venv",
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

    selected_records, dropped_records = select_best_patches(patch_records, model_name)
    model = get_model(model_name)
    input_tensor = prepare_patch_tensors(selected_records, model_name)

    with torch.no_grad():
        if model_name in MIL_MODELS:
            outputs, attention_weights = model(input_tensor, return_attention=True)
            attention_weights = attention_weights[0].detach().cpu().numpy().tolist()
        else:
            outputs = model(input_tensor)
            attention_weights = None

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

    response = {
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
        "patch_importance_supported": model_name in MIL_MODELS,
    }

    if model_name in MIL_MODELS and attention_weights is not None:
        patch_importance = []
        for record, attn in zip(selected_records, attention_weights):
            patch_importance.append({
                "filename": record["filename"],
                "attention_weight": round(float(attn), 6),
                "white_percent": record["white_percent"],
                "tissue_percent": record["tissue_percent"],
            })

        patch_importance.sort(key=lambda x: x["attention_weight"], reverse=True)
        response["patch_importance"] = patch_importance
        response["top_important_patches"] = patch_importance[:10]
    else:
        response["patch_importance"] = None
        response["top_important_patches"] = None

    return response