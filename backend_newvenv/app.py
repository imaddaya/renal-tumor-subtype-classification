from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
import random

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
IMAGE_SIZE = 224
NUM_PATCHES = 70

BASE_DIR = "/home/hpdeadman/Grad_Project/Models"
MODEL_PATHS = {
    "ResNet18 + Vision Mamba": os.path.join(BASE_DIR, "ResNet18_VisionMamba", "results", "ResNet18_VisionMamba_model.pth"),
    "ResNet18 + Vision Mamba + KAN": os.path.join(BASE_DIR, "ResNet18_VisionMamba_KAN", "results", "ResNet18_VisionMamba_KAN_model.pth"),
}

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


def get_top_prediction(probabilities):
    return max(probabilities.items(), key=lambda x: x[1])[0]


class ResNet18VisionMambaModel(nn.Module):
    def __init__(self, num_classes=4, d_state=16, d_conv=4, expand=2):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.mamba = Mamba(d_model=feat_dim, d_state=d_state, d_conv=d_conv, expand=expand)
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
        self.mamba = Mamba(d_model=feat_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.kan = KAN([feat_dim, 128, num_classes])

    def forward(self, x):
        b, n, c, h, w = x.shape
        x = x.view(b * n, c, h, w)
        feats = self.backbone(x)
        feats = feats.view(b, n, -1)
        seq_out = self.mamba(feats)
        slide_feats = seq_out.mean(dim=1)
        return self.kan(slide_feats)


MODELS = {}


def build_model(model_name):
    if model_name == "ResNet18 + Vision Mamba":
        return ResNet18VisionMambaModel(num_classes=NUM_CLASSES)
    if model_name == "ResNet18 + Vision Mamba + KAN":
        return ResNet18VisionMambaKANModel(num_classes=NUM_CLASSES)
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


def prepare_patch_tensors(pil_images):
    patch_tensors = []

    for image in pil_images:
        tensor = transform(image.convert("RGB"))
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
    return {"status": "ok", "device": str(DEVICE), "backend": "newvenv"}


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
    input_tensor = prepare_patch_tensors(pil_images)

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