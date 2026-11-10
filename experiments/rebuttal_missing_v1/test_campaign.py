"""CPU tests for external-method parity, safe submission, and queue dependencies."""
import ast
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import sys
from contextlib import nullcontext

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'code'))
import numpy as np
import torch
from scipy.stats import beta
import runner
from adaptive_cfg import beta_weight, guided_noise, make_forward


def reference_method(path, class_name, method_name):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    method.decorator_list = []
    scope = {'th': torch, 'torch': torch, 'beta': beta}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), scope)
    return scope[method_name]


class CampaignTests(unittest.TestCase):
    def test_beta_matches_author_expression(self):
        original = reference_method(HERE / 'reference/beta_cfg/latent_diffusion.py', 'BetaDDIMGeo', 'get_from_beta')
        torch.manual_seed(7)
        c, u = torch.randn(4, 3, 4, 4), torch.randn(4, 3, 4, 4)
        for progress in np.linspace(0, 1, 50):
            weight = original(None, progress, 2., 3.)
            self.assertAlmostEqual(beta_weight(progress), weight, places=12)
            delta = c - u
            expected = u + 7.5 * delta / torch.norm(delta, p=2, dim=(1, 2, 3)).reshape(-1, 1, 1, 1) * weight
            torch.testing.assert_close(guided_noise(c, u, progress), expected)
        torch.testing.assert_close(guided_noise(c, c, .5), c)

    def test_beta_dit_shape_variance_channels_and_timestep_direction(self):
        class Model:
            def forward(self, x, t, y):
                return torch.arange(len(x)*8*2*2).float().reshape(len(x), 8, 2, 2)
        model = Model()
        mapping = [0, 249, 500, 749, 999]
        forward = make_forward(model, mapping, {'scale': 7.5, 'a': 2., 'b': 3., 'gamma': 1.})
        x, y = torch.zeros(8, 4, 2, 2), torch.zeros(8, dtype=torch.long)
        for original_t, progress in [(999, 0.), (749, .25), (0, 1.)]:
            t = torch.full((8,), original_t)
            raw = model.forward(x, t, y)
            result = forward(x, t, y, 4.)
            self.assertEqual(result.shape, raw.shape)
            torch.testing.assert_close(result[:, 3:], raw[:, 3:])
            torch.testing.assert_close(result[:4, :3], guided_noise(raw[:4, :3], raw[4:, :3], progress))
            torch.testing.assert_close(result[:4, :3], result[4:, :3])

    def test_mgd_official_parity_entire_trajectory_and_boundary(self):
        from ags.ags_sampler import AGSSampler
        original = reference_method(HERE / 'reference/mgd3/gaussian_diffusion.py', 'GaussianDiffusion', 'p_sample_with_mode_guidance')
        class Diffusion:
            num_timesteps = 50
            def p_mean_variance(self, model, x, t, **kwargs):
                return {'mean': .9 * x, 'pred_xstart': .2 * x, 'log_variance': torch.full_like(x, -.5)}
        diffusion = Diffusion()
        model = SimpleNamespace(forward_with_cfg=lambda *args: None)
        for stop in [0, 24, 25, 49]:
            sampler = AGSSampler(model, None, diffusion, device='cpu', use_cags=False, use_iast=False,
                                 use_tags=False, guidance_window='high_noise')
            sampler.per_class_params = {0: {'guidance_strength': .1, 't_stop': stop}}
            z, modes = torch.randn(2, 4, 4, 4), torch.randn(1, 4, 4, 4)
            result = sampler.sample_with_ags_guidance(z.clone(), 0, modes, {}, generator=torch.Generator().manual_seed(42))
            rng = torch.Generator().manual_seed(42)
            reference = z.clone()
            for i in reversed(range(50)):
                noise = torch.randn(reference.shape, generator=rng)
                with patch.object(torch, 'randn_like', return_value=noise):
                    reference = original(diffusion, None, reference, torch.tensor([i, i]), clip_denoised=False,
                                         mode_features=modes, stop_t=stop, mode_guidance_scale=.1)['sample']
            torch.testing.assert_close(result, reference, atol=1e-6, rtol=1e-6)
        self.assertEqual(len([i for i in range(50) if i > 25]), 24)

    def test_counts_dependencies_and_resource_limit(self):
        specs = runner.requests({'account': 'general', 'qos': 'a800'}, 2)
        by_name = {n: (cmd, dep) for n, cmd, dep in specs}
        self.assertEqual(len(set(runner.CONDITIONS)), 18)
        self.assertEqual(by_name['generate'][1], 'plan')
        self.assertIn('--array=0-89%2', by_name['generate'][0])
        self.assertEqual(by_name['evaluate'][1], 'generate')
        self.assertIn('--array=0-14%2', by_name['evaluate'][0])
        self.assertFalse(any('gres' in arg for arg in by_name['profile'][0]))
        self.assertEqual(by_name['profile'][1], 'features')

    def test_submission_ambiguous_response_is_recorded(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(runner, 'HERE', Path(temp)):
            record = {'stages': {}}
            with patch.object(runner.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='unexpected', stderr='')):
                with self.assertRaisesRegex(RuntimeError, 'Unrecognized'):
                    runner.invoke(['sbatch'], record, 'generate')
            self.assertEqual(runner.read(Path(temp) / 'submission.json')['stages']['generate']['status'], 'unknown')

    def test_success_is_persisted_before_next_submission(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(runner, 'HERE', Path(temp)):
            record = {'stages': {}}
            with patch.object(runner.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='12345;cluster\n', stderr='')):
                runner.invoke(['sbatch'], record, 'features')
            self.assertEqual(runner.read(Path(temp) / 'submission.json')['stages']['features']['job_id'], '12345')

    def test_actual_submission_dependencies_and_idempotence(self):
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(status=False, dry_run=False, max_parallel=2, retry_failed=False)
            calls = []
            def fake_submit(command, **kwargs):
                calls.append(command)
                return SimpleNamespace(returncode=0, stdout=str(2000 + len(calls)) + '\n', stderr='')
            with patch.object(runner, 'HERE', Path(temp)), patch.object(runner, 'WORK', Path(temp)/'work'), \
                 patch.object(runner, 'validate', return_value={'campaign_hash': 'test-only'}), \
                 patch.object(runner, 'recover'), patch.object(runner, 'locked', return_value=nullcontext()), \
                 patch.object(runner, 'account_options', return_value={'account': 'general', 'qos': 'a800'}), \
                 patch.object(runner, 'query', return_value=''), patch.object(runner.subprocess, 'run', side_effect=fake_submit), \
                 patch.dict(runner.os.environ, {}, clear=True):
                runner.submit(args)
                self.assertEqual(len(calls), 7)
                for index in range(2, 7):
                    self.assertIn('--dependency=afterok:' + str(2000 + index), calls[index])
                runner.submit(args)
                self.assertEqual(len(calls), 7)

    def test_real_dit_spaced_sampler_accepts_beta_wrapper(self):
        from models import DiT
        from diffusion import create_diffusion
        from ags.ags_sampler import AGSSampler
        torch.set_num_threads(2)
        model = DiT(input_size=4, hidden_size=32, depth=1, num_heads=4).eval()
        # Nonzero output makes this exercise guidance, not only the zero guard.
        torch.nn.init.normal_(model.final_layer.linear.weight, std=.01)
        diffusion = create_diffusion('50')
        model.forward_with_cfg = make_forward(model, diffusion.timestep_map, {'scale': 7.5})
        sampler = AGSSampler(model, None, diffusion, device='cpu', latent_size=4,
                             use_cags=False, use_iast=False, use_tags=False, guidance_window='high_noise')
        sampler.per_class_params = {0: {'guidance_strength': .03, 't_stop': 24}}
        with torch.no_grad():
            result = sampler.sample_with_ags_guidance(torch.randn(8, 4, 4, 4), 0, torch.randn(4, 4, 4, 4),
                {'y': torch.tensor([0]*4 + [1000]*4), 'cfg_scale': 4.}, generator=torch.Generator().manual_seed(9))
        self.assertTrue(torch.isfinite(result).all())

    def test_frozen_plan_construction_in_temporary_directory(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(runner, 'WORK', Path(temp)), \
             patch.object(runner, 'RUN', Path(temp)/'runs/final'), \
             patch.object(runner, 'validate', return_value={'campaign_hash': 'unit-test', 'resnet_sha256': 'unit-test'}):
            parent = runner.read_plan(runner.PARENT / 'plan.json')
            base = runner.read(Path(parent['protocol']['prepared']) / 'profiles/vae_auto.json')
            for name in runner.PROFILES:
                runner.write(Path(temp)/'profiles'/(name + '.json'), dict(base, campaign_hash='unit-test'))
            runner.make_plan()
            plan = runner.read_plan(runner.RUN / 'plan.json')
            self.assertEqual(len(plan['conditions']), 18)
            self.assertEqual(plan['conditions']['mgd3_shared_stop25']['stop_t'], 25)
            self.assertEqual(plan['protocol'], parent['protocol'])
            self.assertEqual(plan['generation_seeds'], [300, 301, 302, 303, 304])
            runner.make_plan()  # Exact rerun must be idempotent.


if __name__ == '__main__':
    unittest.main()
