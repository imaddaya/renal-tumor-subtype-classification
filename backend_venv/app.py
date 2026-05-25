from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
import random
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
IMAGE_SIZE = 224
NUM_PATCHES = 70

BASE_DIR = "/home/hpdeadman/Grad_Project/Models"
MODEL_PATHS = {
    "ResNet18": os.path.join(BASE_DIR, "ResNet18", "results", "ResNet18_model.pth"),
    "ResNet18 + MIL": os.path.join(BASE_DIR, "ResNet18_MIL", "results", "ResNet18_MIL_model.pth"),
    "ResNet18 + MIL + Macenko": os.path.join(BASE_DIR, "ResNet18_MIL_Macenko", "results", "ResNet18_MIL_Macenko_model.pth"),
    "ResNet18 + MIL + KAN": os.path.join(BASE_DIR, "ResNet18_MIL_KAN", "results", "ResNet18_MIL_KAN_model.pth"),
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


def macenko_normalize_pil(pil_img):
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


def get_top_prediction(probabilities):
    return max(probabilities.items(), key=lambda x: x[1])[0]


class SimpleWSIModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        b, n, c, h, w = x.shape
        x = x.view(b * n, c, h, w)
        feats = self.backbone(x)
        feats = feats.view(b, n, -1)
        slide_feats = feats.mean(dim=1)
        return self.classifier(slide_feats)


class AttentionMILModel(nn.Module):
    def __init__(self, num_classes=4, attn_dim=128):
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

    def forward(self, x):
        b, n, c, h, w = x.shape
        x = x.view(b * n, c, h, w)
        feats = self.backbone(x)
        feats = feats.view(b, n, -1)
        attn_scores = self.attention(feats)
        attn_weights = torch.softmax(attn_scores, dim=1)
        slide_feats = torch.sum(attn_weights * feats, dim=1)
        return self.classifier(slide_feats)


class KANLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        self.poly2 = nn.Linear(in_features, out_features, bias=False)
        self.poly3 = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        return self.base(x) + self.poly2(x ** 2) + self.poly3(x ** 3)


class KANClassifier(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes):
        super().__init__()
        self.layer1 = KANLinear(in_features, hidden_features)
        self.norm1 = nn.LayerNorm(hidden_features)
        self.act = nn.GELU()
        self.layer2 = KANLinear(hidden_features, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.layer2(x)
        return x


class AttentionMILKANModel(nn.Module):
    def __init__(self, num_classes=4, attn_dim=128, kan_hidden=128):
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

    def forward(self, x):
        b, n, c, h, w = x.shape
        x = x.view(b * n, c, h, w)
        feats = self.backbone(x)
        feats = feats.view(b, n, -1)
        attn_scores = self.attention(feats)
        attn_weights = torch.softmax(attn_scores, dim=1)
        slide_feats = torch.sum(attn_weights * feats, dim=1)
        return self.classifier(slide_feats)


MODELS = {}


def build_model(model_name):
    if model_name == "ResNet18":
        return SimpleWSIModel(num_classes=NUM_CLASSES)
    if model_name == "ResNet18 + MIL":
        return AttentionMILModel(num_classes=NUM_CLASSES)
    if model_name == "ResNet18 + MIL + Macenko":
        return AttentionMILModel(num_classes=NUM_CLASSES)
    if model_name == "ResNet18 + MIL + KAN":
        return AttentionMILKANModel(num_classes=NUM_CLASSES)
    raise ValueError(f"Unsupported model: {model_name}")


def get_model(model_name):
    if model_name in MODELS:
        return MODELS[model_name]

    model_path = MODEL_PATHS.get(model_name)
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found for {model_name}: {model_path}")

    model = build_model(model_name).to(DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    MODELS[model_name] = model
    return model


def prepare_patch_tensors(pil_images, model_name):
    patch_tensors = []

    for image in pil_images:
        image = image.convert("RGB")

        if model_name == "ResNet18 + MIL + Macenko":
            try:
                image = macenko_normalize_pil(image)
            except Exception:
                pass

        tensor = transform(image)
        patch_tensors.append(tensor)

    if len(patch_tensors) == 0:
        raise ValueError("No valid patch images were provided.")

    if len(patch_tensors) > NUM_PATCHES:
        patch_tensors = patch_tensors[:NUM_PATCHES]
    elif len(patch_tensors) < NUM_PATCHES:
        needed = NUM_PATCHES - len(patch_tensors)
        extra = random.choices(patch_tensors, k=needed)
        patch_tensors.extend(extra)

    patches = torch.stack(patch_tensors, dim=0)
    return patches.unsqueeze(0).to(DEVICE)


@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "backend": "venv"}


@app.get("/models")
def list_models():
    return {"models": list(MODEL_PATHS.keys())}


@app.post("/predict")
async def predict(
    model_name: str = Form(...),
    true_label: str = Form(...),
    images: list[UploadFile] = File(...),
):
    if model_name not in MODEL_PATHS:
        return {"error": f"Unknown model: {model_name}"}

    if true_label not in CLASS_NAMES:
        return {"error": f"Unknown true label: {true_label}"}

    if not images:
        return {"error": "No images were uploaded."}

    pil_images = []
    for uploaded in images:
        image_bytes = await uploaded.read()
        pil_images.append(Image.open(io.BytesIO(image_bytes)).convert("RGB"))

    model = get_model(model_name)
    input_tensor = prepare_patch_tensors(pil_images, model_name)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0].detach().cpu().numpy().tolist()

    probabilities = {CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)}
    predicted_label = get_top_prediction(probabilities)

    return {
        "model": model_name,
        "true_label": true_label,
        "predicted_label": predicted_label,
        "correct": predicted_label == true_label,
        "patch_count_uploaded": len(pil_images),
        "patch_count_used": NUM_PATCHES,
        "probabilities": probabilities,
    }