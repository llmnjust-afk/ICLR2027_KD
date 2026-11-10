"""Format compatibility through real Parquet extraction and CPU data loaders."""
import io
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image

from rebuttal import data_setup, resources
from rebuttal.common import atomic_json, file_hash
from rebuttal.prepare import make_split, SourceImages
from rebuttal.evaluate import evaluation_paths, EvaluationImages


class ImageFormatTests(unittest.TestCase):
    def fixture(self, tmp, *, corrupt=False):
        target = Path(tmp) / 'dataset'
        cls = 'n13037406'
        inputs = []
        for format_name, mode, color in [('TIFF', 'CMYK', (12, 34, 56, 78)), ('PNG', 'L', 80),
                                         ('JPEG', 'RGB', (50, 60, 70)), ('BMP', 'RGB', (1, 2, 3)),
                                         ('WEBP', 'RGB', (5, 6, 7)), ('PNG', 'RGB', (10, 20, 30))]:
            buffer = io.BytesIO()
            Image.new(mode, (24, 20), color).save(buffer, format=format_name)
            inputs.append(buffer.getvalue())
        if corrupt:
            inputs[0] = b'not an image'
        manifest = {'label_to_wnid': {'0': cls}, 'manifest_hash': 'fixture'}
        pairs = []
        for filename, images in [(data_setup.SHARDS[0], [inputs[0]] + [inputs[2]] * 49),
                                  (data_setup.SHARDS[1], inputs * 2)]:
            shard = Path(tmp) / Path(filename).name
            pq.write_table(pa.Table.from_pydict({'image': [{'bytes': raw, 'path': None} for raw in images],
                                                'label': [0] * len(images)}), shard)
            item = {'filename': filename, 'size': shard.stat().st_size, 'sha256': file_hash(shard)}
            pairs.append((item, shard))
            data_setup.extract(target, shard, item, manifest)
        return target, cls, inputs, manifest, pairs

    def test_cmyk_tiff_and_grayscale_preserve_bytes_and_work_in_train_and_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            target, cls, inputs, manifest, pairs = self.fixture(tmp)
            for item, shard in pairs:
                counts, _ = data_setup.check_records(target, item, manifest)
                self.assertEqual(counts[cls], 50 if 'validation-' in item['filename'] else 12)
            first = target / 'train' / cls / 'train-00000-of-00017_00000000.tiff'
            gray = target / 'train' / cls / 'train-00000-of-00017_00000001.png'
            self.assertEqual(first.read_bytes(), inputs[0])
            self.assertEqual(gray.read_bytes(), inputs[1])
            split = make_split(target, [cls], .25, 2027, 2)
            all_paths = split['train'][cls] + split['tune'][cls]
            self.assertEqual(len(all_paths), 12)
            self.assertIn(str(first.relative_to(target)), all_paths)
            self.assertEqual(sum(p.endswith('.tiff') for p in all_paths), 2)
            data = SourceImages(target, all_paths, image_size=16)
            for index in range(len(data)):
                tensor, rel, checksum = data[index]
                self.assertEqual(tuple(tensor.shape), (3, 16, 16))
                self.assertTrue(torch.isfinite(tensor).all())
                self.assertEqual(checksum, file_hash(target / rel))
            prepared = Path(tmp) / 'prepared'
            atomic_json(prepared / 'split.json', split)
            root, paths = evaluation_paths(prepared, target / 'val', 'report')
            self.assertEqual(len(paths), 50)
            self.assertTrue(any(path.endswith('.tiff') for path, label in paths))
            evaluation = EvaluationImages(root, paths, size=16)
            for index, (rel, _) in enumerate(paths):
                if rel.endswith('.tiff'):
                    tensor, label = evaluation[index]
                    self.assertEqual(tuple(tensor.shape), (3, 16, 16))
                    self.assertTrue(torch.isfinite(tensor).all())
                    self.assertEqual(label, 0)
            inventory = resources.dataset_check(target, [cls], ipc=2)
            self.assertTrue(inventory['valid'])
            self.assertEqual(inventory['counts'][cls], {'train': 12, 'val': 50})

    def test_corrupt_image_reports_shard_row_and_wnid_and_leaves_no_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, r'shard=data/validation-.*row=0, wnid=n13037406; format='):
                self.fixture(tmp, corrupt=True)
            self.assertFalse(list((Path(tmp) / 'dataset/download_records').glob('*.done.json')))

    def test_existing_tif_alias_is_included_by_file_scanners(self):
        with tempfile.TemporaryDirectory() as tmp:
            target, cls, inputs, manifest, pairs = self.fixture(tmp)
            path = target / 'train' / cls / 'external.tif'
            path.write_bytes(inputs[0])
            split = make_split(target, [cls], .25, 2027, 2)
            self.assertIn(str(path.relative_to(target)), split['train'][cls] + split['tune'][cls])
            self.assertEqual(resources.dataset_check(target, [cls], ipc=2)['counts'][cls]['train'], 13)


if __name__ == '__main__':
    unittest.main()
