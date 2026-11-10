import contextlib
import io
import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest import mock

from rebuttal import campaign, slurm
from rebuttal.common import atomic_json, file_hash, read_json
from rebuttal.reporting import runtime_estimate


class CampaignTests(unittest.TestCase):
    def test_runtime_estimate_uses_matching_completed_runs_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = {'plan_hash': 'p', 'generation_seeds': [100, 101], 'training_seeds': [0, 1, 2],
                    'architectures': ['convnet6'], 'conditions': {'cags': {}, 'no_mode': {}}}
            atomic_json(root / 'datasets/g100/cags/manifest.json', {'plan_hash': 'p', 'elapsed_s': 3600})
            atomic_json(root / 'datasets/g100/no_mode/manifest.json', {'plan_hash': 'old', 'elapsed_s': 999999})
            atomic_json(root / 'evaluation/g100/t0/convnet6/cags.json',
                        {'plan_hash': 'p', 'elapsed_s': 1800, 'metrics': {'top1': 20}})
            estimate = runtime_estimate(root, plan)
            self.assertEqual(estimate['expected_fits'], 12)
            self.assertEqual(estimate['measured_datasets'], 1)
            self.assertEqual(estimate['estimated_remaining_gpu_hours'], 8.5)
            self.assertEqual(estimate['ideal_remaining_wall_hours'], 2.125)

    def test_dry_run_and_status_never_read_a_submission_pointer(self):
        for mode in ['dry_run', 'status']:
            args = types.SimpleNamespace(dry_run=mode == 'dry_run', status=mode == 'status', retry_failed=False)
            with mock.patch.object(campaign, 'command') as command, \
                    mock.patch.object(campaign, 'read_json', side_effect=AssertionError('no pointer needed')), \
                    contextlib.redirect_stdout(io.StringIO()):
                campaign.run(args)
            self.assertEqual(command.call_count, 2)
            self.assertTrue(all('--' + mode.replace('_', '-') in call.args[0] for call in command.call_args_list))

    def test_submission_failure_never_launches_second_batch(self):
        with mock.patch.object(campaign, 'command', side_effect=subprocess.CalledProcessError(1, ['submit'])) as command:
            with self.assertRaises(subprocess.CalledProcessError), contextlib.redirect_stdout(io.StringIO()):
                campaign.run(types.SimpleNamespace(dry_run=False, status=False, retry_failed=True))
        self.assertEqual(command.call_count, 1)

    def test_second_batch_uses_first_batch_record_and_distinct_config(self):
        with mock.patch.object(campaign, 'command') as command, \
                mock.patch.object(campaign, 'read_json', return_value={'batch': '/first-batch'}), \
                contextlib.redirect_stdout(io.StringIO()):
            campaign.run(types.SimpleNamespace(dry_run=False, status=False, retry_failed=True))
        first, second = command.call_args_list
        self.assertEqual(second.args[0][-2:], ['--after-batch', '/first-batch'])
        self.assertNotEqual(first.kwargs['env']['REBUTTAL_CONFIG'], second.kwargs['env']['REBUTTAL_CONFIG'])

    def test_prepared_reuse_validates_split_order_and_preserves_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, work, data = root / 'source', root / 'work', root / 'data'
            source.mkdir(); work.mkdir(); data.mkdir()
            classes = root / 'classes.txt'
            classes.write_text('n00000001\n')
            atomic_json(source / 'split.json', {'root': str(data)})
            manifest = {'config': {'nclass': 1, 'ipc': 10, 'holdout_fraction': .1, 'seed': 2027},
                        'classes': ['n00000001'], 'files': {'split.json': file_hash(source / 'split.json')}}
            atomic_json(source / 'manifest.json', manifest)
            before = file_hash(source / 'manifest.json')
            env = {'NCLASS': '1', 'IPC': '10', 'HOLDOUT_FRACTION': '.1', 'PREPARE_SEED': '2027',
                   'DATA_ROOT': str(data), 'CLASS_FILE': str(classes)}
            with mock.patch.dict(os.environ, env):
                slurm.reuse_prepared(work, source)
                slurm.reuse_prepared(work, source)
                self.assertEqual((work / 'prepared').resolve(), source)
                self.assertEqual(file_hash(source / 'manifest.json'), before)
                self.assertEqual(read_json(work / 'cache_reuse.json')['verified_files'], 1)
                classes.write_text('n99999999\n')
                with self.assertRaisesRegex(ValueError, '顺序'):
                    slurm.reuse_prepared(work, source)


if __name__ == '__main__':
    unittest.main()
