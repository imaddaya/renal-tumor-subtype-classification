"""CPU API boundaries; no dataset or trained checkpoint is required."""

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from fastapi.testclient import TestClient
from PIL import Image
from backend_venv import app as api


def uploads(count=70):
    output = io.BytesIO()
    Image.new("RGB", (32, 32), (150, 70, 100)).save(output, format="PNG")
    return [
        ("images", (f"patch-{n}.png", output.getvalue(), "image/png"))
        for n in range(count)
    ]


class InferenceApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)
        api.MODELS.clear()
        api.get_normalizer.cache_clear()

    def test_health_does_not_require_reference_image(self):
        with patch.object(api, "TARGET_IMAGE_PATH", Path("/missing-reference.png")):
            response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model_num_patches"], 70)

    def test_upload_count_is_validated(self):
        response = self.client.post(
            "/predict",
            data={"model_name": "ResNet18", "true_label": "clearcell"},
            files=uploads(1),
        )
        self.assertEqual(response.status_code, 400)

    def test_missing_weights_return_actionable_error(self):
        with patch.dict(
            api.MODEL_PATH_CANDIDATES, {"ResNet18": ["/missing-model.pth"]}
        ):
            response = self.client.post(
                "/predict",
                data={"model_name": "ResNet18", "true_label": "clearcell"},
                files=uploads(),
            )
        self.assertEqual(response.status_code, 503)
        self.assertIn("RCC_MODELS_DIR", response.json()["detail"])

    def test_missing_reference_does_not_skip_normalization(self):
        with patch.object(
            api, "TARGET_IMAGE_PATH", Path("/missing-reference.png")
        ), patch.object(api, "get_model", return_value=object()):
            response = self.client.post(
                "/predict",
                data={
                    "model_name": "ResNet18 + MIL + Macenko",
                    "true_label": "clearcell",
                },
                files=uploads(),
            )
        self.assertEqual(response.status_code, 503)
        self.assertIn("RCC_MACENKO_REFERENCE", response.json()["detail"])

    def test_normalization_failure_does_not_run_unnormalized_inference(self):
        with patch.object(api, "get_model", return_value=object()), patch.object(
            api, "macenko_normalize_pil", side_effect=ValueError("bad patch")
        ):
            response = self.client.post(
                "/predict",
                data={
                    "model_name": "ResNet18 + MIL + Macenko",
                    "true_label": "clearcell",
                },
                files=uploads(),
            )
        self.assertEqual(response.status_code, 422)

    def test_non_macenko_request_runs_without_reference(self):
        class TinyModel(torch.nn.Module):
            def forward(self, patches):
                assert patches.shape == (1, 70, 3, 224, 224)
                assert not torch.is_grad_enabled()
                return torch.zeros((1, 4), device=patches.device)

        with patch.object(
            api, "TARGET_IMAGE_PATH", Path("/missing-reference.png")
        ), patch.object(api, "get_model", return_value=TinyModel()):
            response = self.client.post(
                "/predict",
                data={"model_name": "ResNet18", "true_label": "clearcell"},
                files=uploads(),
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["patch_count_used"], 70)
        self.assertAlmostEqual(sum(response.json()["probabilities"].values()), 1)

    def test_macenko_normalization_uses_configured_reference(self):
        # Synthetic pixels exercise numerical code, not scientific accuracy.
        pixels = np.random.default_rng(42).integers(
            50, 220, (64, 64, 3), dtype=np.uint8
        )
        with tempfile.TemporaryDirectory() as folder:
            reference = Path(folder) / "reference.png"
            Image.fromarray(pixels).save(reference)
            with patch.object(api, "TARGET_IMAGE_PATH", reference):
                result = api.macenko_normalize_pil(Image.fromarray(pixels))
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.size, (64, 64))
