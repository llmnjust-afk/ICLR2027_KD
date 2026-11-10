"""Scientific parity, exact scope, resumption and scheduler safety tests."""
import ast
from contextlib import nullcontext
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import runner as r


def args(**changes):
    return SimpleNamespace(**dict(dict(status=False, dry_run=False, max_parallel=None, retry_failed=False), **changes))


class LauncherTests(unittest.TestCase):
    def test_scope_and_resource_requests(self):
        for parallel in [1, 2]:
            requests = r.scheduler_requests(parallel, {'account': 'general', 'qos': None})
            self.assertEqual([x[0] for x in requests], ['smoke', 'generate', 'evaluate', 'summarize'])
            self.assertIn(f'--array=0-4%{parallel}', requests[1][1])
            self.assertIn(f'--array=0-14%{parallel}', requests[2][1])
            for _, command, _ in requests[:3]:
                self.assertIn('--partition=rtx4090', command)
                self.assertIn('--gres=gpu:4090:1', command)
                self.assertFalse(any('a800' in word.lower() for word in command))
            self.assertFalse(any('gres' in word for word in requests[-1][1]))

    def test_submission_dependencies_idempotence_and_inherited_parallelism(self):
        with tempfile.TemporaryDirectory() as temp:
            calls = []
            def fake(command, **kw):
                calls.append(command)
                return SimpleNamespace(returncode=0, stdout=f'{6000 + len(calls)};cluster\n', stderr='')
            with patch.object(r, 'HERE', Path(temp)), patch.object(r, 'WORK', Path(temp)/'work'), \
                 patch.object(r, 'validate', return_value={'plan_hash': 'test'}), \
                 patch.object(r, 'recover'), patch.object(r, 'locked', return_value=nullcontext()), \
                 patch.object(r, 'query', return_value=''), patch.object(r.subprocess, 'run', side_effect=fake), \
                 patch.dict(r.os.environ, {}, clear=True):
                r.submit(args(max_parallel=1))
                self.assertEqual(len(calls), 4)
                for index in range(1, 4):
                    self.assertIn('--dependency=afterok:' + str(6000 + index), calls[index])
                self.assertIn('--array=0-14%1', calls[2])
                r.submit(args())  # Persist one-GPU preference across invocations.
                self.assertEqual(len(calls), 4)
                with self.assertRaisesRegex(ValueError, 'different settings'):
                    r.submit(args(max_parallel=2))
                self.assertEqual(len(calls), 4)

    def test_failed_preflight_creates_no_jobs(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(r, 'HERE', Path(temp)), \
             patch.object(r, 'WORK', Path(temp)/'work'), patch.object(r, 'validate', return_value={'plan_hash': 'test'}), \
             patch.object(r, 'recover'), patch.object(r, 'locked', return_value=nullcontext()), \
             patch.object(r, 'query', side_effect=RuntimeError('bad resource request')), \
             patch.object(r, 'invoke') as invoke, patch.dict(r.os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'bad resource'):
                r.submit(args())
            invoke.assert_not_called()
            self.assertFalse((Path(temp)/'submission.json').exists())

    def test_ambiguous_submission_is_recorded_and_not_repeated(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(r, 'HERE', Path(temp)):
            record = {'plan_hash': 'test', 'max_parallel': 2, 'options': {'account': 'general', 'qos': None}, 'stages': {}}
            with patch.object(r.subprocess, 'run', side_effect=r.subprocess.TimeoutExpired('sbatch', 45)):
                with self.assertRaisesRegex(RuntimeError, 'unknown'):
                    r.invoke(['sbatch'], record, 'generate')
            self.assertEqual(r.read(Path(temp)/'submission.json')['stages']['generate']['status'], 'unknown')
            with patch.object(r, 'validate', return_value={'plan_hash': 'test'}), patch.object(r, 'recover'), \
                 patch.object(r, 'locked', return_value=nullcontext()), patch.object(r, 'invoke') as invoke, \
                 patch.dict(r.os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, 'unknown'):
                    r.submit(args())
                invoke.assert_not_called()

    def test_retry_rejects_live_or_unknown_jobs(self):
        record = {'stages': {'evaluate': {'job_id': '6003'}}}
        for output in ['', '6003|RUNNING|\n', '6003_0|COMPLETED|\n6003_1|PENDING|\n']:
            with patch.object(r, 'query', return_value=output):
                with self.assertRaises(RuntimeError):
                    r.require_terminal(record)
        with patch.object(r, 'query', return_value='6003_0|COMPLETED|\n6003_1|FAILED|\n'):
            r.require_terminal(record)

    def test_original_official_mgd3_trajectory(self):
        import torch
        from ags.ags_sampler import AGSSampler
        path = r.HERE / 'reference/gaussian_diffusion.py'
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GaussianDiffusion')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'p_sample_with_mode_guidance')
        scope = {'th': torch}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), scope)
        official = scope[method.name]
        class Diffusion:
            num_timesteps = 50
            def p_mean_variance(self, model, x, t, **kw):
                return {'mean': .9*x, 'pred_xstart': .2*x, 'log_variance': torch.full_like(x, -.5)}
        model = SimpleNamespace(forward_with_cfg=lambda *a: None)
        diffusion = Diffusion()
        sampler = AGSSampler(model, None, diffusion, device='cpu', use_cags=False, use_iast=False,
                             use_tags=False, guidance_window='high_noise')
        sampler.per_class_params = {0: {'guidance_strength': .1, 't_stop': 25}}
        z, modes = torch.randn(2, 4, 4, 4), torch.randn(1, 4, 4, 4)
        actual = sampler.sample_with_ags_guidance(z.clone(), 0, modes, {}, generator=torch.Generator().manual_seed(42))
        rng, expected = torch.Generator().manual_seed(42), z.clone()
        for t in reversed(range(50)):
            with patch.object(torch, 'randn_like', return_value=torch.randn(expected.shape, generator=rng)):
                expected = official(diffusion, None, expected, torch.tensor([t, t]),
                    clip_denoised=False, mode_features=modes, stop_t=25, mode_guidance_scale=.1)['sample']
        torch.testing.assert_close(actual, expected)
        self.assertEqual(len([t for t in range(50) if t > 25]), 24)

    def test_parent_core_snapshot_and_seed_grid(self):
        parent = r.read_plan(r.PARENT/'plan.json')
        self.assertEqual(r.source_manifest(), parent['source'])
        self.assertEqual(parent['generation_seeds'], [100, 101, 102, 103, 104])
        self.assertEqual(parent['training_seeds'], [0, 1, 2])
        self.assertEqual(parent['protocol']['epochs'], 2000)
        self.assertEqual(parent['protocol']['batch_size'], 128)
        self.assertEqual(parent['protocol']['generation_batch_size'], 4)

    def test_import_identity_detects_mixed_baseline(self):
        parent = r.read_plan(r.PARENT/'plan.json')
        path = r.result_path(r.PARENT, 100, 0, 'cags')
        result = r.check_result(path, parent, 100, 0, 'cags')
        with tempfile.TemporaryDirectory() as temp:
            bad = Path(temp)/'mixed.json'
            r.write(bad, dict(result, train_seed=99))
            with self.assertRaisesRegex(ValueError, 'train_seed'):
                r.check_result(bad, parent, 100, 0, 'cags')
            altered = copy.deepcopy(result)
            altered['hardware']['gpu'] = 'NVIDIA A800 80GB PCIe'
            r.write(bad, altered)
            with self.assertRaisesRegex(ValueError, 'hardware'):
                r.check_result(bad, parent, 100, 0, 'cags')

    def test_evaluator_trains_only_new_mgd_condition(self):
        from rebuttal import run as original_run
        from rebuttal import evaluate, generate
        parent = r.read_plan(r.PARENT/'plan.json')
        plan = copy.deepcopy(parent)
        plan['plan_hash'] = 'unit-test-not-scientific'
        plan['conditions'] = {k: parent['conditions'][k] for k in r.BASELINES}
        plan['conditions'][r.METHOD] = {'coefficients': [.1]*100, 'stop_t': 25}
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            r.write(folder/'evaluation_images.json', r.read(r.PARENT/'evaluation_images.json'))
            for name in r.BASELINES:
                result = r.read(r.result_path(r.PARENT, 100, 0, name))
                result['plan_hash'] = plan['plan_hash']
                r.write(r.result_path(folder, 100, 0, name), result)
            with patch.object(original_run, 'hardware', return_value={'gpu': r.GPU}), \
                 patch.object(evaluate, 'image_manifest', return_value=plan['protocol']['evaluation_image_manifest']), \
                 patch.object(evaluate, 'loader', return_value=object()), \
                 patch.object(generate, 'verify_dataset', return_value={}), \
                 patch.object(original_run, 'file_hash', return_value='test-only-hash'), \
                 patch.object(evaluate, 'train_one') as training:
                original_run.execute_evaluation(plan, folder, {'generation_seed': 100, 'training_seed': 0, 'architecture': 'convnet6'})
                self.assertEqual(training.call_count, 1)
                self.assertEqual(training.call_args.args[0], folder/'datasets/g100'/r.METHOD)


if __name__ == '__main__':
    unittest.main()
