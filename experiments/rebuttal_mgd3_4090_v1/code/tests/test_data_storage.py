"""Real filesystem migration tests; no live dataset or network is touched."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rebuttal.common import atomic_json, file_hash
from rebuttal.data_setup import REPO, REVISION, dataset_cache
from rebuttal import storage


class StorageTests(unittest.TestCase):
    def fixture(self, tmp):
        resources = Path(tmp) / 'project/resources'
        source = resources / 'datasets/imagenet100'
        target = Path(tmp) / 'ImageNet100'
        target.mkdir()
        atomic_json(source / 'download_source.json', {'repo': REPO, 'revision': REVISION})
        image = source / 'train/n02869837/fixture.JPEG'
        image.parent.mkdir(parents=True)
        image.write_bytes(b'original image bytes')
        namespace = 'datasets--' + REPO.replace('/', '--')
        blob = resources / 'hf_cache' / namespace / 'blobs/test-blob'
        blob.parent.mkdir(parents=True)
        blob.write_bytes(b'original cached shard')
        (blob.parent / 'partial.curl.incomplete').write_bytes(b'partial bytes')
        vae = resources / 'hf_cache/models--test--vae/weights'
        vae.parent.mkdir(parents=True)
        vae.write_bytes(b'unchanged model')
        (resources / 'core_transfer_events.jsonl').write_text('{"event": "previous download"}\n')
        return resources, source, target, namespace, image, blob, vae

    def migrate(self, resources, target):
        storage.migrate_legacy_layout(resources, target, target / '.hf_cache', repo=REPO, revision=REVISION)

    def test_migration_preserves_hashes_partial_files_aliases_and_model_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            resources, source, target, namespace, image, blob, vae = self.fixture(tmp)
            before = [file_hash(p) for p in (image, blob, vae)]
            self.migrate(resources, target)
            self.assertEqual([file_hash(p) for p in (image, blob, vae)], before)
            self.assertTrue(source.is_symlink())
            self.assertEqual(source.resolve(), target.resolve())
            self.assertTrue((resources / 'hf_cache' / namespace).is_symlink())
            self.assertFalse(vae.parent.is_symlink())
            self.assertEqual((target / '.hf_cache' / namespace / 'blobs/partial.curl.incomplete').read_bytes(), b'partial bytes')
            self.assertTrue((resources / 'core_transfer_events.jsonl').is_symlink())
            self.assertTrue((target / 'core_transfer_events.jsonl').exists())
            self.migrate(resources, target)
            self.assertEqual([file_hash(p) for p in (image, blob, vae)], before)
            self.assertEqual(json.loads((resources / 'imagenet100_storage_migration.json').read_text())['state'], 'complete')

    def test_conflicting_content_stops_before_moving_anything(self):
        with tempfile.TemporaryDirectory() as tmp:
            resources, source, target, namespace, image, blob, vae = self.fixture(tmp)
            (target / 'download_source.json').write_bytes((source / 'download_source.json').read_bytes())
            conflicting = target / image.relative_to(source)
            conflicting.parent.mkdir(parents=True)
            conflicting.write_bytes(b'other data, never overwrite')
            with self.assertRaises(ValueError):
                self.migrate(resources, target)
            self.assertFalse(source.is_symlink())
            self.assertTrue(blob.exists())
            self.assertEqual(conflicting.read_bytes(), b'other data, never overwrite')
            self.assertFalse((resources / 'imagenet100_storage_migration.json').exists())

    def test_unregistered_nonempty_destination_and_wrong_revision_rejected(self):
        for registered in (False, True):
            with self.subTest(registered=registered), tempfile.TemporaryDirectory() as tmp:
                resources, source, target, _, image, blob, _ = self.fixture(tmp)
                if registered:
                    atomic_json(target / 'download_source.json', {'repo': REPO, 'revision': 'wrong'})
                else:
                    (target / 'unrelated.txt').write_text('user data')
                with self.assertRaises(ValueError):
                    self.migrate(resources, target)
                self.assertTrue(image.exists())
                self.assertTrue(blob.exists())
                self.assertFalse(source.is_symlink())

    def test_interruption_after_move_before_alias_creation_is_resumable(self):
        with tempfile.TemporaryDirectory() as tmp:
            resources, source, target, _, image, blob, _ = self.fixture(tmp)
            original_link = Path.symlink_to
            def interrupt(path, destination, **kwargs):
                if path == source:
                    raise OSError('simulated interruption after move')
                return original_link(path, destination, **kwargs)
            with patch.object(Path, 'symlink_to', interrupt), self.assertRaises(OSError):
                self.migrate(resources, target)
            self.assertFalse(source.exists())
            self.assertTrue((target / 'download_source.json').exists())
            self.migrate(resources, target)
            self.assertTrue(source.is_symlink())
            self.assertEqual(image.read_bytes(), b'original image bytes')
            self.assertEqual(blob.read_bytes(), b'original cached shard')

    def test_identical_existing_files_are_merged_without_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            resources, source, target, _, image, _, _ = self.fixture(tmp)
            for rel in ('download_source.json', 'train/n02869837/fixture.JPEG'):
                path = target / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((source / rel).read_bytes())
            self.migrate(resources, target)
            self.assertTrue(source.is_symlink())
            self.assertEqual(len(list((target / 'train').rglob('*.JPEG'))), 1)

    def test_cache_configuration_overrides_default(self):
        with patch.dict(os.environ, {'REBUTTAL_DATA_CACHE': ''}):
            self.assertEqual(dataset_cache(Path('/data/root')), Path('/data/root/.hf_cache'))
        with patch.dict(os.environ, {'REBUTTAL_DATA_CACHE': '/data/custom-cache'}):
            self.assertEqual(dataset_cache(Path('/data/root')), Path('/data/custom-cache'))


if __name__ == '__main__':
    unittest.main()
