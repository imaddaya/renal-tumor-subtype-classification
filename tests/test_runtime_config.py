import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import runtime_config as config


class RuntimeConfigurationTests(unittest.TestCase):
    def test_default_paths_work_from_another_directory(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as folder, patch.dict(
            os.environ, {}, clear=True
        ):
            try:
                os.chdir(folder)
                self.assertEqual(config.models_dir(), config.PROJECT_ROOT / "Models")
                self.assertTrue(config.macenko_reference_path().is_absolute())
            finally:
                os.chdir(previous)

    def test_relative_override_is_relative_to_repository(self):
        with patch.dict(
            os.environ,
            {
                "RCC_MODELS_DIR": "external/weights",
                "RCC_MACENKO_REFERENCE": "assets/reference.png",
            },
        ):
            self.assertEqual(
                config.models_dir(), config.PROJECT_ROOT / "external/weights"
            )
            self.assertEqual(
                config.macenko_reference_path(),
                config.PROJECT_ROOT / "assets/reference.png",
            )

    def test_absolute_override_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(
            os.environ, {"RCC_MODELS_DIR": folder}
        ):
            self.assertEqual(config.models_dir(), Path(folder))

    def test_cors_origins_are_normalized(self):
        with patch.dict(
            os.environ,
            {
                "RCC_ALLOWED_ORIGINS": " http://localhost:5173/, https://example.test/ , "
            },
        ):
            self.assertEqual(
                config.cors_origins(), ["http://localhost:5173", "https://example.test"]
            )
