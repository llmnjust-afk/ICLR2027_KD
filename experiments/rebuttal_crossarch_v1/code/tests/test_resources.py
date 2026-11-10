"""Offline resource tests with explicit toy images; no benchmark/model downloads."""
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from rebuttal import resources


class ResourceTests(unittest.TestCase):
    def test_unknown_label_order_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Refusing to assume"):
            resources.resolve_row_wnid({"label": 0, "image": {"path": "validation_0.JPEG"}}, None, {}, {})

    def test_wnid_mapping_checks_filename_metadata_agreement(self):
        aliases = resources.imagenet_aliases()
        inferred = {}
        row = {"label": 0, "image": {"path": "n01749939_100.JPEG"}}
        self.assertEqual(resources.resolve_row_wnid(row, ["n01749939"], inferred, aliases), "n01749939")
        self.assertEqual(inferred["0"], "n01749939")
        with self.assertRaisesRegex(ValueError, "uniquely"):
            resources.resolve_row_wnid(row, ["n02869837"], {}, aliases)
        self.assertEqual(resources.metadata_names({"info": {"features": {"label": {"names": ["n01749939"]}}}}), ["n01749939"])

    def test_partial_class_dataset_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ["train", "val"]:
                (root / split / "n01749939").mkdir(parents=True)
                for i in range(5):
                    (root / split / "n01749939" / f"{i}.jpg").write_bytes(b"toy presence check")
            check = resources.dataset_check(root, ["n01749939", "n02869837"], ipc=2)
            self.assertFalse(check["valid"])
            self.assertIn("train/n02869837", check["missing"])

    def test_partial_managed_download_is_not_used_as_complete_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ['train', 'val']:
                folder = root / split / 'n01749939'
                folder.mkdir(parents=True)
                for i in range(5):
                    (folder / f'{i}.jpg').write_bytes(b'presence fixture')
            (root / 'download_source.json').write_text('{}')
            self.assertFalse(resources.dataset_check(root, ['n01749939'], ipc=2)['valid'])
            self.assertTrue(resources.dataset_check(root, ['n01749939'], ipc=2, allow_unfinished=True)['valid'])

    def test_download_extract_reversed_labels_and_resume(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("PyArrow is optional for this offline extraction test")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = ["n01749939", "n02869837"]
            files = {}
            for split, count in [("train", 6), ("validation", 50)]:
                rows = []
                for label, wnid in enumerate(names):
                    for index in range(count):
                        buffer = io.BytesIO()
                        Image.new("RGB", (8, 8), (label * 100, index, 40)).save(buffer, format="PNG")
                        rows.append({"label": label, "image": {"bytes": buffer.getvalue(),
                                     "path": f"{wnid}_{index}.png" if split == "train" else f"ILSVRC_val_{label}_{index}.png"}})
                table = pa.Table.from_pydict({"label": [r["label"] for r in rows],
                                            "image": [r["image"] for r in rows]})
                metadata = {b"huggingface": json.dumps({"info": {"features": {
                    "label": {"_type": "ClassLabel", "names": names}}}}).encode()}
                table = table.replace_schema_metadata(metadata)
                filename = f"data/{split}-00000-of-00001.parquet"
                local = root / f"{split}.parquet"
                pq.write_table(table, local)
                files[filename] = str(local)
            api = types.SimpleNamespace(dataset_info=lambda *a, **k: types.SimpleNamespace(sha="toy-revision", card_data=None),
                                        list_repo_files=lambda *a, **k: list(files))
            def fake_download(repo, filename, **kwargs):
                self.assertEqual(kwargs["revision"], "toy-revision")
                return files[filename]
            # Requested class-file order is intentionally the reverse of the dataset label order.
            expected = list(reversed(names))
            with patch("huggingface_hub.HfApi", return_value=api), patch("huggingface_hub.hf_hub_download", side_effect=fake_download):
                data, report = resources.ensure_dataset(root / "resources", [], expected, 2)
            saved = json.loads((Path(data) / "resource_manifest.json").read_text())
            self.assertEqual(saved["label_to_wnid"]["0"], "n01749939")
            self.assertEqual(saved["counts"]["n01749939"], {"train": 6, "val": 50})
            with Image.open(next((Path(data) / "train/n01749939").glob("*.png"))) as image:
                self.assertEqual(image.getpixel((0, 0))[0], 0)
            with patch("huggingface_hub.HfApi", return_value=api), patch("huggingface_hub.hf_hub_download") as downloader:
                resources.ensure_dataset(root / "resources", [], expected, 2)
                downloader.assert_not_called()

    def test_config_update_preserves_existing_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config.env"
            config.write_text("GPUS=2,3\nEPOCHS=2000\n")
            with patch.object(resources, "ROOT", root):
                resources.write_config(config, {"DATA_ROOT": "/data/path with spaces"})
            self.assertIn("GPUS=2,3", config.read_text())
            self.assertIn("DATA_ROOT='/data/path with spaces'", config.read_text())
            self.assertEqual(len(list((root / "backups").glob("*.env"))), 1)


if __name__ == "__main__":
    unittest.main()
