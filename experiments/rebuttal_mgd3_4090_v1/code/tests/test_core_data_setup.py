"""Offline integration tests; tiny Parquets exercise the real extraction/verification code."""
import hashlib
import io
import json
import os
import socket
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from rebuttal import data_setup as setup, network
from rebuttal.common import ROOT, atomic_json, digest, file_hash
from rebuttal.transfers import DownloadError


def response(code=0, status=200, **kwargs):
    return dict(curl_exit=code, http_status=status, error='', content_range=None,
                limit_exceeded=False, **kwargs)


class NetworkTests(unittest.TestCase):
    def test_proxy_socket_permission_failure_is_recorded_without_credentials(self):
        with patch.dict(os.environ, {'https_proxy': 'http://secret:password@127.0.0.1:19992'}), \
                patch.object(socket, 'create_connection', side_effect=PermissionError('Operation not permitted')):
            result = network.proxy_probe()
        self.assertFalse(result['reachable'])
        self.assertEqual(result['error_type'], 'PermissionError')
        self.assertNotIn('secret', json.dumps(result))

    def test_tls12_still_verifies_certificates_and_ignores_curlrc(self):
        with patch.object(network.shutil, 'which', return_value='/usr/bin/curl'):
            cmd = network.curl_command('1.2')
        self.assertEqual(cmd[:2], ['curl', '-q'])
        self.assertNotIn('--insecure', cmd)
        self.assertNotIn('-k', cmd)
        self.assertIn('--tls-max', cmd)
        self.assertIn('--tlsv1.2', cmd)

    def test_error_redacts_proxy_credentials_and_signed_cdn_url(self):
        text = network.sanitize('TLS EOF https://user:secret@proxy.example/path?token=abc '
                                'https://cdn.example/file?Signature=private-value')
        self.assertIn('TLS EOF', text)
        self.assertIn('cdn.example', text)
        self.assertNotIn('secret', text)
        self.assertNotIn('private-value', text)
        self.assertNotIn('abc', text)

    def test_curl_records_real_redirect_host_and_bounds_probe(self):
        def launch(command, stdout, stderr):
            self.assertEqual(command[command.index('--range') + 1], '0-65535')
            self.assertEqual(command[command.index('--max-filesize') + 1], '65536')
            self.assertNotIn('--continue-at', command)
            Path(command[command.index('--output') + 1]).write_bytes(b'PAR1')
            Path(command[command.index('--dump-header') + 1]).write_text(
                'HTTP/1.1 302 Found\r\nLocation: https://cdn.example/data?Signature=secret\r\n'
                'HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 0-65535/100000\r\n')
            stdout.write(json.dumps({'http_code': 206, 'url_effective': 'https://cdn.example/data?Signature=secret'}))
            return types.SimpleNamespace(returncode=0, poll=lambda: 0, wait=lambda **kw: 0)
        with tempfile.TemporaryDirectory() as tmp, patch.object(network.subprocess, 'Popen', side_effect=launch):
            result = network.curl_request('https://example.com/data', Path(tmp) / 'probe',
                                          byte_range='0-65535', limit=65536)
        self.assertEqual(result['effective_host'], 'cdn.example')
        self.assertEqual(result['redirect_hosts'], ['cdn.example'])
        self.assertEqual(result['content_range'], 'bytes 0-65535/100000')
        self.assertNotIn('secret', json.dumps(result))

    def transfer_fixture(self, tmp, raw=b'PAR1 fixture payload'):
        item = {'filename': setup.SHARDS[1], 'size': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        target, partial = network.blob_paths(tmp, setup.REPO, item['sha256'])
        partial.parent.mkdir(parents=True)
        return item, target, partial

    def test_resume_reuses_existing_partial_and_promotes_only_verified_bytes(self):
        data = b'PAR1 fixture payload'
        with tempfile.TemporaryDirectory() as tmp:
            item, target, partial = self.transfer_fixture(tmp, data)
            partial.write_bytes(data[:7])
            def request(url, output, **kwargs):
                self.assertEqual(output.read_bytes(), data[:7])
                self.assertTrue(kwargs['transfer'])
                with output.open('ab') as stream:
                    stream.write(data[7:])
                return response()
            with patch.object(network, 'curl_request', side_effect=request) as curl:
                got = network.download(item, repo=setup.REPO, revision=setup.REVISION, cache=tmp,
                                       routes=[{'endpoint': 'https://example.com', 'tls': 'default'}])
            self.assertEqual(got, target)
            self.assertEqual(target.read_bytes(), data)
            self.assertFalse(partial.exists())
            self.assertEqual(curl.call_count, 1)

    def test_verified_cached_blob_and_complete_partial_need_no_network(self):
        for complete_partial in (False, True):
            with self.subTest(partial=complete_partial), tempfile.TemporaryDirectory() as tmp:
                item, target, partial = self.transfer_fixture(tmp)
                (partial if complete_partial else target).write_bytes(b'PAR1 fixture payload')
                with patch.object(network, 'curl_request', side_effect=AssertionError('network called')):
                    got = network.download(item, repo=setup.REPO, revision=setup.REVISION, cache=tmp, routes=[])
                self.assertEqual(got, target)

    def test_stalled_transfer_stops_after_two_attempts_and_preserves_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            item, target, partial = self.transfer_fixture(tmp)
            partial.write_bytes(b'PAR1')
            with patch.dict(os.environ, {'REBUTTAL_DOWNLOAD_RETRIES': '9'}), \
                    patch.object(network, 'curl_request', return_value=response(35, 0)) as curl:
                with self.assertRaises(DownloadError):
                    network.download(item, repo=setup.REPO, revision=setup.REVISION, cache=tmp,
                                     routes=[{'endpoint': 'https://example.com', 'tls': 'default'}])
            self.assertEqual(curl.call_count, 2)
            self.assertEqual(partial.read_bytes(), b'PAR1')
            self.assertFalse(target.exists())

    def test_corrupt_complete_transfer_is_quarantined_never_marked_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            item, target, partial = self.transfer_fixture(tmp)
            def corrupt(url, output, **kwargs):
                output.write_bytes(b'X' * item['size'])
                return response()
            with patch.dict(os.environ, {'REBUTTAL_DOWNLOAD_RETRIES': '1'}), \
                    patch.object(network, 'curl_request', side_effect=corrupt):
                with self.assertRaises(DownloadError):
                    network.download(item, repo=setup.REPO, revision=setup.REVISION, cache=tmp,
                                     routes=[{'endpoint': 'https://example.com', 'tls': 'default'}])
            self.assertFalse(target.exists())
            self.assertTrue(list(partial.parent.glob(partial.name + '.bad_hash_*')))


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.classes = (ROOT / 'misc/class100.txt').read_text().splitlines()
        ids = (ROOT / 'misc/class_indices.txt').read_text().splitlines()
        names = (ROOT / 'misc/class_names.txt').read_text().splitlines()
        self.source_names = dict(zip(ids, names))

    def manifest(self):
        sizes = [1] * 17 + [setup.TOTAL_BYTES - 17]
        info = {'sha': setup.REVISION, 'siblings': [
            {'rfilename': filename, 'lfs': {'size': size, 'sha256': hashlib.sha256(filename.encode()).hexdigest()}}
            for filename, size in zip(setup.SHARDS, sizes)]}
        names = {str(i): self.source_names[c] for i, c in enumerate(reversed(self.classes))}
        return setup.metadata_manifest(info, self.classes, names)

    def test_class_mapping_uses_published_names_even_if_class_file_order_differs(self):
        value = self.manifest()
        self.assertEqual(value['label_to_wnid']['0'], self.classes[-1])
        self.assertNotEqual(value['label_to_wnid']['0'], self.classes[0])
        self.assertEqual(value['files'][0]['filename'], setup.SHARDS[0])

    def test_ambiguous_labels_and_wrong_class_set_rejected(self):
        with self.assertRaises(ValueError):
            setup.label_mapping(['unknown'] * 100, self.classes)
        bad = self.classes[:-1] + ['n00000000']
        with self.assertRaises(ValueError):
            setup.validate_manifest(self.manifest(), bad)

    def test_wrong_revision_size_and_digest_fail_closed(self):
        for key in ('revision', 'size', 'manifest_hash'):
            value = self.manifest()
            if key == 'revision':
                value['revision'] = 'a' * 40
            elif key == 'size':
                value['files'][0]['size'] += 1
                value['manifest_hash'] = digest({k: v for k, v in value.items() if k != 'manifest_hash'})
            else:
                value['manifest_hash'] = 'invalid'
            with self.subTest(key=key), self.assertRaises(ValueError):
                setup.validate_manifest(value, self.classes)

    def test_literal_class_mapping_does_not_execute_source(self):
        value = {c: self.source_names[c] for c in self.classes}
        code = "raise RuntimeError('never execute this')\nclasses = OrderedDict(" + repr(value) + ")"
        self.assertEqual(setup.source_class_dict(code), value)

    def test_api_and_real_cdn_range_probe_both_required_and_tls_fallback_recorded(self):
        manifest = self.manifest()
        range_response = response(0, 206)
        range_response['content_range'] = f"bytes 0-65535/{manifest['files'][0]['size']}"
        range_response['redirect_hosts'] = ['cdn.example']
        body = b'PAR1' + b'\0' * (65536 - 4)
        replies = [(b'', response(35, 0)), (json.dumps({'sha': setup.REVISION}).encode(), response()),
                   (body, range_response)]
        with tempfile.TemporaryDirectory() as tmp, patch.object(network, 'proxy_probe', return_value={}), \
                patch.object(network, 'endpoints', return_value=['https://example.com']), \
                patch.object(network, 'fetch_small', side_effect=replies) as fetch:
            result, routes = setup.diagnose(tmp, self.classes, manifest)
            saved = json.loads((Path(tmp) / 'core_network_diagnostics.json').read_text())
        self.assertEqual(routes, [{'endpoint': 'https://example.com', 'tls': '1.2'}])
        self.assertEqual(fetch.call_args_list[0].kwargs['tls'], 'default')
        self.assertEqual(fetch.call_args_list[1].kwargs['tls'], '1.2')
        self.assertEqual(fetch.call_args_list[2].kwargs['byte_range'], '0-65535')
        self.assertIn('cdn.example', saved['checks'][-1]['redirect_hosts'])

    def test_working_api_but_cdn_tls_failure_does_not_start_downloads(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(network, 'proxy_probe', return_value={}), \
                patch.object(network, 'endpoints', return_value=['https://example.com']), \
                patch.object(network, 'fetch_small', side_effect=[
                    (json.dumps({'sha': setup.REVISION}).encode(), response()), (b'', response(35, 0)),
                    (json.dumps({'sha': setup.REVISION}).encode(), response()), (b'', response(35, 0))]), \
                patch.object(network, 'download') as download:
            with self.assertRaises(DownloadError):
                setup.diagnose(tmp, self.classes, self.manifest())
            report = json.loads((Path(tmp) / 'core_network_diagnostics.json').read_text())
        self.assertFalse(report['ready_to_download'])
        download.assert_not_called()

    def test_cached_manifest_allows_cdn_download_route_when_api_is_unavailable(self):
        manifest = self.manifest()
        good = response(0, 206)
        good['content_range'] = f"bytes 0-65535/{manifest['files'][0]['size']}"
        with tempfile.TemporaryDirectory() as tmp, patch.object(network, 'proxy_probe', return_value={}), \
                patch.object(network, 'endpoints', return_value=['https://example.com']), \
                patch.object(network, 'fetch_small', side_effect=[(b'', response(22, 503)),
                                                                  (b'PAR1' + bytes(65532), good)]):
            _, routes = setup.diagnose(tmp, self.classes, manifest)
        self.assertEqual(len(routes), 1)

    def test_successful_first_download_caches_complete_manifest(self):
        value = self.manifest()
        info = {'sha': setup.REVISION, 'siblings': [
            {'rfilename': f['filename'], 'lfs': {'size': f['size'], 'sha256': f['sha256']}}
            for f in value['files']], 'cardData': {'dataset_info': {'features': [
                {'name': 'label', 'dtype': {'class_label': {'names': value['label_names']}}}]}}}
        good = response(0, 206)
        good['content_range'] = f"bytes 0-65535/{value['files'][0]['size']}"
        with tempfile.TemporaryDirectory() as tmp, patch.object(network, 'proxy_probe', return_value={}), \
                patch.object(network, 'endpoints', return_value=['https://example.com']), \
                patch.object(network, 'fetch_small', side_effect=[(json.dumps(info).encode(), response()),
                                                                  (b'PAR1' + bytes(65532), good)]):
            saved, _ = setup.diagnose(tmp, self.classes)
            self.assertEqual(json.loads(setup.manifest_path(tmp).read_text()), saved)
        self.assertEqual(len(saved['files']), 18)


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {'REBUTTAL_DATA_CACHE': '', 'REPORT_DIR': ''})
        env.start()
        self.addCleanup(env.stop)

    def fixture(self, resources):
        resources = Path(resources)
        raw = io.BytesIO()
        Image.new('RGB', (8, 8), (10, 20, 30)).save(raw, format='JPEG')
        raw = raw.getvalue()
        target = resources / 'data'
        manifest = {'label_to_wnid': {'0': 'n02869837'}, 'manifest_hash': 'fixture',
                    'files': [], 'revision': setup.REVISION}
        for filename, count in [(setup.SHARDS[0], 50), (setup.SHARDS[1], 1)]:
            shard = resources / Path(filename).name
            pq.write_table(pa.Table.from_pydict({'image': [{'bytes': raw, 'path': None}] * count,
                                               'label': [0] * count}), shard)
            item = {'filename': filename, 'size': shard.stat().st_size, 'sha256': file_hash(shard)}
            blob, _ = network.blob_paths(setup.dataset_cache(target), setup.REPO, item['sha256'])
            blob.parent.mkdir(parents=True, exist_ok=True)
            shard.replace(blob)
            manifest['files'].append(item)
            setup.extract(target, blob, item, manifest)
        return target, manifest, raw

    def test_preserves_image_bytes_and_full_offline_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            target, manifest, raw = self.fixture(tmp)
            with patch.object(network, 'curl_request', side_effect=AssertionError('network called')), \
                    patch.object(setup, 'EXPECTED_ROWS', {'train': 1, 'val': 50}):
                counts = setup.verify_dataset(tmp, target, manifest, ['n02869837'])
            self.assertEqual(counts, {'n02869837': {'train': 1, 'val': 50}})
            self.assertEqual(next((target / 'val/n02869837').iterdir()).read_bytes(), raw)

    def test_corrupt_image_is_detected_then_repaired_from_verified_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            target, manifest, raw = self.fixture(tmp)
            image = next((target / 'val/n02869837').iterdir())
            image.write_bytes(b'broken')
            with self.assertRaises(ValueError):
                setup.check_records(target, manifest['files'][0], manifest)
            blob, _ = network.blob_paths(setup.dataset_cache(target), setup.REPO, manifest['files'][0]['sha256'])
            setup.extract(target, blob, manifest['files'][0], manifest)
            setup.check_records(target, manifest['files'][0], manifest)
            self.assertEqual(image.read_bytes(), raw)
            self.assertEqual(len(list((target / 'corrupt_images').rglob('*.JPEG'))), 1)

    def test_record_label_corruption_rejected_even_if_record_hash_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            target, manifest, _ = self.fixture(tmp)
            marker, record = setup.marker_paths(target, manifest['files'][0])
            rows = [json.loads(line) for line in record.read_text().splitlines()]
            rows[0]['label'] = 1
            record.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            saved = json.loads(marker.read_text())
            saved['record_sha256'] = file_hash(record)
            atomic_json(marker, saved)
            with self.assertRaises(ValueError):
                setup.check_records(target, manifest['files'][0], manifest)

    def test_unrecorded_images_cannot_silently_change_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            target, manifest, raw = self.fixture(tmp)
            (target / 'val/n02869837/extra.jpg').write_bytes(raw)
            with patch.object(setup, 'EXPECTED_ROWS', {'train': 1, 'val': 50}), self.assertRaises(ValueError):
                setup.verify_dataset(tmp, target, manifest, ['n02869837'])

    def test_extra_empty_class_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            target, manifest, _ = self.fixture(tmp)
            (target / 'val/n00000000').mkdir()
            with patch.object(setup, 'EXPECTED_ROWS', {'train': 1, 'val': 50}), self.assertRaises(ValueError):
                setup.verify_dataset(tmp, target, manifest, ['n02869837'])

    def test_missing_manifest_offline_verify_does_not_call_network_or_create_ready_marker(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(setup, 'check_models'), \
                patch.object(network, 'curl_request', side_effect=AssertionError('network called')), \
                patch.dict(os.environ, {'REPORT_DIR': ''}):
            target = Path(tmp) / 'data'
            with self.assertRaises(ValueError):
                setup.run('verify', tmp, target, ['n02869837'])
            self.assertFalse((target / 'resource_manifest.json').exists())

    def test_prepare_and_verify_complete_offline_and_extra_file_revokes_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            target, manifest, raw = self.fixture(tmp)
            manifest.update(repo=setup.REPO, image_policy='test fixture bytes')
            atomic_json(setup.manifest_path(tmp), manifest)
            atomic_json(target / 'download_source.json', {'repo': setup.REPO, 'revision': setup.REVISION})
            # Inject the tiny protocol; the real 100-class/18-shard manifest validator
            # is tested separately above. All extraction/hash/decode paths remain real.
            with patch.object(setup, 'validate_manifest', return_value=manifest), \
                    patch.object(setup, 'check_models'), patch.object(setup, 'EXPECTED_ROWS', {'train': 1, 'val': 50}), \
                    patch.dict(os.environ, {'REPORT_DIR': ''}), \
                    patch.object(network, 'fetch_small', side_effect=AssertionError('network called')):
                setup.run('prepare', tmp, target, ['n02869837'])
                self.assertTrue((target / 'resource_manifest.json').exists())
                setup.run('verify', tmp, target, ['n02869837'])
                (target / 'val/n02869837/extra.jpg').write_bytes(raw)
                with self.assertRaises(ValueError):
                    setup.run('prepare', tmp, target, ['n02869837'])
                self.assertFalse((target / 'resource_manifest.json').exists())

    def test_cached_shards_are_processed_before_network_failure_on_next_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            target, manifest, raw = self.fixture(tmp)
            manifest.update(repo=setup.REPO, image_policy='test fixture bytes')
            missing = {'filename': setup.SHARDS[2], 'size': 100, 'sha256': 'a' * 64}
            manifest['files'].append(missing)
            atomic_json(setup.manifest_path(tmp), manifest)
            atomic_json(target / 'download_source.json', {'repo': setup.REPO, 'revision': setup.REVISION})
            # Force the second cached shard to be re-extracted, without network access.
            marker, record = setup.marker_paths(target, manifest['files'][1])
            marker.unlink()
            for image in (target / 'train').rglob('*.JPEG'):
                image.unlink()
            def failed_diagnosis(*args):
                self.assertTrue(marker.exists())
                self.assertEqual(sum(setup.check_records(target, manifest['files'][1], manifest)[0].values()), 1)
                raise DownloadError('fixture: proxy unavailable')
            with patch.object(setup, 'validate_manifest', return_value=manifest), \
                    patch.object(setup, 'check_models'), patch.object(setup, 'require_space'), \
                    patch.object(setup, 'diagnose', side_effect=failed_diagnosis) as diagnose:
                with self.assertRaises(DownloadError):
                    setup.run('prepare', tmp, target, ['n02869837'])
            diagnose.assert_called_once()
            self.assertFalse((target / 'resource_manifest.json').exists())


if __name__ == '__main__':
    unittest.main()
