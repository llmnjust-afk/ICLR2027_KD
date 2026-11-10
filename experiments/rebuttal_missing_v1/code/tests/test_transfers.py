"""Exercise interrupted/corrupt transfers without downloading benchmark artifacts."""
import hashlib
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from huggingface_hub.utils import LocalEntryNotFoundError
from requests.exceptions import SSLError

from rebuttal import transfers


class TransferTests(unittest.TestCase):
    def test_incomplete_head_uses_pinned_repository_lfs_metadata(self):
        revision, checksum = 'a' * 40, 'b' * 64
        # Mirrors can omit any of these resolver headers, or expose a non-LFS ETag.
        for etag, size, commit in [(None, None, None), ('proxy-etag', 128, revision),
                                    (checksum, None, revision), (checksum, 123, None)]:
            with self.subTest(etag=etag, size=size, commit=commit):
                head = types.SimpleNamespace(etag=etag, size=size, commit_hash=commit)
                info = types.SimpleNamespace(sha=revision, siblings=[types.SimpleNamespace(
                    rfilename='data.parquet', lfs={'sha256': checksum, 'size': 123})])
                with patch('huggingface_hub.get_hf_file_metadata', return_value=head) as lookup, \
                        patch('huggingface_hub.HfApi') as api:
                    api.return_value.repo_info.return_value = info
                    result = transfers.artifact_metadata('test/data', 'data.parquet', repo_type='dataset',
                                                         revision=revision, endpoint='https://example.com')
                self.assertEqual(result, (checksum, 123))
                self.assertNotIn('?', lookup.call_args.args[0])
                self.assertEqual(api.return_value.repo_info.call_args.kwargs['revision'], revision)
                self.assertTrue(api.return_value.repo_info.call_args.kwargs['files_metadata'])

    def test_missing_head_and_api_metadata_use_raw_lfs_pointer(self):
        revision, checksum = 'a' * 40, 'b' * 64
        info = types.SimpleNamespace(sha=revision, siblings=[])
        pointer = f'version https://git-lfs.github.com/spec/v1\noid sha256:{checksum}\nsize 123\n'.encode()
        with patch('huggingface_hub.get_hf_file_metadata', side_effect=SSLError('HEAD unavailable')), \
                patch('huggingface_hub.HfApi') as api, patch('requests.get') as get:
            api.return_value.repo_info.return_value = info
            get.return_value.__enter__.return_value.iter_content.return_value = iter([pointer])
            result = transfers.artifact_metadata('test/data', 'data.parquet', repo_type='dataset',
                                                 revision=revision, endpoint='https://example.com')
        self.assertEqual(result, (checksum, 123))
        self.assertIn('/raw/' + revision + '/', get.call_args.args[0])
        self.assertTrue(get.call_args.kwargs['stream'])

    def test_unverifiable_metadata_is_retryable_and_does_not_disable_hash_check(self):
        info = types.SimpleNamespace(sha='wrong-revision', siblings=[types.SimpleNamespace(
            rfilename='data.parquet', lfs={'sha256': 'b' * 64, 'size': 123})])
        with patch('huggingface_hub.get_hf_file_metadata', side_effect=SSLError('HEAD unavailable')), \
                patch('huggingface_hub.HfApi') as api, patch('requests.get') as get:
            api.return_value.repo_info.return_value = info
            get.return_value.__enter__.return_value.iter_content.return_value = iter([b'<html>proxy error</html>'])
            with self.assertRaises(transfers.DownloadError):
                transfers.artifact_metadata('test/data', 'data.parquet', repo_type='dataset',
                                             revision='a' * 40, endpoint='https://example.com')

    def test_metadata_failure_on_mirror_continues_to_official_endpoint(self):
        data = b'verified shard fixture'
        checksum = hashlib.sha256(data).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            def transfer(command, **kwargs):
                self.assertIn('https://huggingface.co/', command[-1])
                Path(command[command.index('--output') + 1]).write_bytes(data)
                return types.SimpleNamespace(returncode=0, stderr='')
            with patch.object(transfers, 'endpoints', return_value=['https://hf-mirror.com', 'https://huggingface.co']), \
                    patch.object(transfers, 'artifact_metadata', side_effect=[
                        transfers.DownloadError('mirror omitted metadata'), (checksum, len(data))]) as meta, \
                    patch.object(transfers.shutil, 'which', return_value='/usr/bin/curl'), \
                    patch.object(transfers.subprocess, 'run', side_effect=transfer):
                path = transfers.curl_download('test/data', 'data.parquet', repo_type='dataset',
                                               revision='a' * 40, cache_dir=tmp)
            self.assertEqual(Path(path).read_bytes(), data)
            self.assertEqual(meta.call_count, 2)

    def test_ssl_failure_uses_curl_after_cache_lookup(self):
        with patch.dict(os.environ, {'REBUTTAL_DOWNLOAD_TRANSPORT': 'auto'}), \
                patch('huggingface_hub.hf_hub_download', side_effect=[
                    LocalEntryNotFoundError('uncached'), SSLError('TLS EOF')]) as hub, \
                patch.object(transfers, 'curl_download', return_value='/verified/blob') as curl:
            result = transfers.hub_download('test/data', 'data.parquet', repo_type='dataset',
                                            revision='abc', cache_dir='/tmp/cache')
        self.assertEqual(result, '/verified/blob')
        self.assertTrue(hub.call_args_list[0].kwargs['local_files_only'])
        self.assertIs(hub.call_args_list[1].kwargs['token'], False)
        curl.assert_called_once()

    def test_cached_file_never_uses_network(self):
        with patch('huggingface_hub.hf_hub_download', return_value='/cached') as hub, \
                patch.object(transfers, 'curl_download') as curl:
            self.assertEqual(transfers.hub_download('test/data', 'file', revision='abc', cache_dir='/tmp'), '/cached')
        hub.assert_called_once()
        self.assertTrue(hub.call_args.kwargs['local_files_only'])
        curl.assert_not_called()

    def test_interrupted_transfer_resumes_and_verifies_digest(self):
        data = b'complete benchmark shard fixture'
        checksum = hashlib.sha256(data).hexdigest()
        revision = 'a' * 40
        metadata = types.SimpleNamespace(etag=checksum, size=len(data), commit_hash=revision)
        with tempfile.TemporaryDirectory() as tmp:
            calls = []

            def transfer(command, **kwargs):
                self.assertIn('--http1.1', command)
                self.assertIn('--continue-at', command)
                self.assertNotIn('--insecure', command)
                output = Path(command[command.index('--output') + 1])
                calls.append(command[-1])
                if len(calls) == 1:
                    output.write_bytes(data[:10])
                    return types.SimpleNamespace(returncode=56, stderr='TLS EOF')
                self.assertEqual(output.read_bytes(), data[:10])
                with output.open('ab') as stream:
                    stream.write(data[10:])
                return types.SimpleNamespace(returncode=0, stderr='')

            with patch.dict(os.environ, {'REBUTTAL_DOWNLOAD_RETRIES': '2'}), \
                    patch.object(transfers, 'endpoints', return_value=['https://example.com']), \
                    patch('huggingface_hub.get_hf_file_metadata', return_value=metadata), \
                    patch.object(transfers.shutil, 'which', return_value='/usr/bin/curl'), \
                    patch.object(transfers.subprocess, 'run', side_effect=transfer), \
                    patch.object(transfers.time, 'sleep'):
                path = transfers.curl_download('test/data', 'data.parquet', repo_type='dataset',
                                               revision=revision, cache_dir=tmp)
            self.assertEqual(Path(path).read_bytes(), data)
            self.assertEqual(Path(path).name, checksum)
            self.assertNotEqual(calls[0], calls[1])
            self.assertFalse(Path(path + '.curl.incomplete').exists())

    def test_wrong_digest_never_becomes_completed_blob(self):
        data = b'correct'
        checksum = hashlib.sha256(data).hexdigest()
        metadata = types.SimpleNamespace(etag=checksum, size=len(data), commit_hash='a' * 40)
        with tempfile.TemporaryDirectory() as tmp:
            def corrupt(command, **kwargs):
                Path(command[command.index('--output') + 1]).write_bytes(b'incorrc')
                return types.SimpleNamespace(returncode=0, stderr='')
            with patch.dict(os.environ, {'REBUTTAL_DOWNLOAD_RETRIES': '1'}), \
                    patch.object(transfers, 'endpoints', return_value=['https://example.com']), \
                    patch('huggingface_hub.get_hf_file_metadata', return_value=metadata), \
                    patch.object(transfers.shutil, 'which', return_value='/usr/bin/curl'), \
                    patch.object(transfers.subprocess, 'run', side_effect=corrupt):
                with self.assertRaises(transfers.DownloadError):
                    transfers.curl_download('test/data', 'data.parquet', repo_type='dataset',
                                             revision='a' * 40, cache_dir=tmp)
            target = Path(tmp) / 'datasets--test--data/blobs' / checksum
            self.assertFalse(target.exists())
            self.assertTrue(Path(str(target) + '.curl.incomplete').exists())


if __name__ == '__main__':
    unittest.main()
