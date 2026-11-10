"""CPU checks for the experiment controls, queue, statistics, and evaluation contract."""
import contextlib
import copy
import io
import json
import multiprocessing
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ags.ags_sampler import AGSSampler
from diffusion.gaussian_diffusion import GaussianDiffusion
from rebuttal.common import atomic_json, coefficient_vectors, condition_set, digest, file_hash, source_manifest
from rebuttal.prepare import make_split, profile_from_features
from rebuttal.run import (bind_generation_gpu, claim_job, mean_matched, parser,
                          plan_experiment, select_fixed)
from rebuttal.statistics import paired_stats, summarize


def queue_contender(job, barrier, out):
    barrier.wait()
    lock = claim_job(Path(job))
    out.put(lock is not None)


def toy_stats():
    names = [f"n{i:08d}" for i in range(8)]
    return {"classes": names, "k_max": 20, "per_class": {
        cls: {"dispersion": i + 1., "entropy_normalized": .15 + i * .1,
              "k": 2 + i % 2, "separability": .2 + .03 * i} for i, cls in enumerate(names)}}


class FakeDiffusion:
    num_timesteps = 4

    def p_mean_variance(self, model, image, time, **kwargs):
        return {"mean": image * .8, "pred_xstart": image * .4,
                "log_variance": torch.zeros_like(image) - 2}


class ControlTests(unittest.TestCase):
    def test_split_is_stable_disjoint_and_keeps_class_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            classes = ["b", "a"]
            for cls in classes:
                (root / "train" / cls).mkdir(parents=True)
                for i in range(20):
                    (root / "train" / cls / f"{i}.jpg").write_bytes(b"fixture")
            one = make_split(root, classes, .2, 2027, 3)
            self.assertEqual(one, make_split(root, classes, .2, 2027, 3))
            for cls in classes:
                self.assertFalse(set(one["train"][cls]) & set(one["tune"][cls]))
                self.assertEqual(len(one["train"][cls]), 16)
                self.assertEqual(len(one["tune"][cls]), 4)
            self.assertEqual(one["classes"], classes)

    def test_permutations_preserve_histogram_and_mean_matches(self):
        controls = condition_set(toy_stats(), "full", [.02], [17, 29, 43], .03)
        original = controls["cags"]["coefficients"]
        self.assertAlmostEqual(np.mean(controls["fixed_mean"]["coefficients"]), np.mean(original))
        for key in ["permuted_17", "permuted_29", "permuted_43"]:
            self.assertEqual(sorted(original), sorted(controls[key]["coefficients"]))
            self.assertNotEqual(original, controls[key]["coefficients"])
        matched = mean_matched([.001, .02, .05], float(np.mean(original)))
        self.assertAlmostEqual(np.mean(matched), np.mean(original), places=12)
        self.assertTrue(all(0 < x < .06 for x in matched))
        self.assertEqual(list(np.argsort(matched)), [0, 1, 2])

    def test_allocation_controls_need_no_tuning_and_preserve_distribution(self):
        stats = toy_stats()
        controls = condition_set(stats, "allocation", [.1], [17, 29, 43])
        self.assertEqual(set(controls), {"no_mode", "cags", "fixed_mean",
                                         "permuted_17", "permuted_29", "permuted_43"})
        values = controls["cags"]["coefficients"]
        self.assertEqual(controls["no_mode"]["coefficients"], [0.] * len(values))
        self.assertTrue(all(x == np.mean(values) for x in controls["fixed_mean"]["coefficients"]))
        for seed in [17, 29, 43]:
            control = controls[f"permuted_{seed}"]
            self.assertEqual(control["permutation_of"], "cags")
            self.assertEqual(sorted(control["coefficients"]), sorted(values))
            self.assertNotEqual(control["coefficients"], values)
            self.assertEqual(control["coefficients"], [values[i] for i in control["source_class_indices"]])
        for seeds in [[], [17, 17]]:
            with self.assertRaisesRegex(ValueError, "nonempty and unique"):
                condition_set(stats, "allocation", [], seeds)

    def sampler(self, coefficient):
        model = types.SimpleNamespace(forward_with_cfg=lambda *args, **kwargs: None)
        sampler = AGSSampler(model, None, FakeDiffusion(), device="cpu", num_sampling_steps=4,
                             use_tags=False, use_iast=False, use_cags=False, guidance_window="high_noise")
        sampler.per_class_params = {0: {"guidance_strength": coefficient, "t_stop": 1}}
        return sampler

    def test_paired_stream_consumption_and_global_rng_isolation(self):
        z, target = torch.ones(2, 4, 2, 2), torch.zeros(1, 4, 2, 2)
        generator1 = torch.Generator().manual_seed(11)
        generator2 = torch.Generator().manual_seed(11)
        before = torch.get_rng_state().clone()
        a = self.sampler(0).sample_with_ags_guidance(z, 0, target, {"cfg_scale": 4}, generator1)
        b = self.sampler(.03).sample_with_ags_guidance(z, 0, target, {"cfg_scale": 4}, generator2)
        self.assertTrue(torch.equal(generator1.get_state(), generator2.get_state()))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertFalse(torch.equal(a, b))

    def test_sampler_regression_against_inherited_mgd_kernel(self):
        z, target = torch.ones(2, 4, 2, 2), torch.zeros(1, 4, 2, 2)
        sampler = self.sampler(.1)
        torch.manual_seed(77)
        expected = z.clone()
        for step in reversed(range(4)):
            out = GaussianDiffusion.p_sample_with_mode_guidance(
                sampler.diffusion, None, expected, torch.tensor([step, step]),
                clip_denoised=False, model_kwargs={"cfg_scale": 4}, mode_features=target,
                stop_t=1, mode_guidance_scale=.1)
            expected = out["sample"]
        torch.manual_seed(77)
        actual = sampler.sample_with_ags_guidance(z, 0, target, {"cfg_scale": 4})
        self.assertTrue(torch.equal(expected, actual))

    def test_cfg_policy_does_not_modify_mode_coefficient(self):
        sampler = self.sampler(.03)
        observed = []
        original = sampler.diffusion.p_mean_variance
        def capture(*args, **kwargs):
            observed.append(kwargs["model_kwargs"]["cfg_scale"])
            return original(*args, **kwargs)
        sampler.diffusion.p_mean_variance = capture
        sampler.sample_with_ags_guidance(torch.ones(2, 4, 2, 2), 0, torch.zeros(1, 4, 2, 2),
                                         {"cfg_scale": 4}, torch.Generator().manual_seed(1),
                                         lambda step, total, base: base if step > 1 else 1)
        self.assertEqual(observed, [4, 4, 1, 1])
        self.assertEqual(sampler.per_class_params[0]["guidance_strength"], .03)

    def test_batched_guidance_pairs_each_sample_with_its_target(self):
        sampler = self.sampler(.1)
        sampler.diffusion.num_timesteps = 1
        sampler.num_sampling_steps = 1
        sampler.per_class_params[0]["t_stop"] = -1
        image = torch.ones(6, 4, 2, 2)
        targets = torch.arange(3.).view(3, 1, 1, 1).expand(3, 4, 2, 2)
        delta = -(image[:3] * .4 - targets) * .1
        expected = image * .8 + torch.cat([delta, delta]) * np.exp(-1.)
        actual = sampler.sample_with_ags_guidance(image, 0, targets, {"cfg_scale": 4},
                                                  torch.Generator().manual_seed(8))
        self.assertTrue(torch.allclose(actual, expected))

    def test_queue_claim_has_one_owner_across_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.job.json"
            atomic_json(path, {"stage": "generate"})
            ctx = multiprocessing.get_context("fork")
            barrier, out = ctx.Barrier(4), ctx.Queue()
            children = [ctx.Process(target=queue_contender, args=(str(path), barrier, out)) for _ in range(4)]
            for process in children:
                process.start()
            winners = [out.get(timeout=10) for _ in children]
            for process in children:
                process.join(timeout=10)
                self.assertEqual(process.exitcode, 0)
            self.assertEqual(sum(winners), 1)
            atomic_json(Path(tmp) / "a.done.json", {})
            (Path(tmp) / "a.lock").unlink()
            self.assertIsNone(claim_job(path))

    def test_generation_gpu_binding_is_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertTrue(bind_generation_gpu(root, 100, {"gpu": "4090"}))
            self.assertFalse(bind_generation_gpu(root, 100, {"gpu": "A800"}))
            self.assertTrue(bind_generation_gpu(root, 101, {"gpu": "A800"}))

    def test_generation_level_ci_uses_generation_count(self):
        result = paired_stats([1, 2, 3])
        self.assertEqual(result["n_generation"], 3)
        self.assertEqual(result["mean_pp"], 2)
        self.assertAlmostEqual(result["sd_pp"], 1)
        self.assertLess(result["ci95_pp"][0], 0)
        self.assertIsNone(paired_stats([2])["p_two_sided"])


class PipelineTests(unittest.TestCase):
    def test_generation_manifests_resume_and_share_noise(self):
        from rebuttal.generate import GenerationEngine, verify_dataset
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            target_file = root / "targets.npz"
            np.savez(target_file, a=np.zeros((3, 4096), dtype=np.float32), b=np.ones((3, 4096), dtype=np.float32))
            engine = GenerationEngine.__new__(GenerationEngine)
            engine.device = torch.device("cpu")
            engine.model = types.SimpleNamespace(forward_with_cfg=lambda *args, **kwargs: None)
            engine.diffusion = FakeDiffusion()
            engine.vae = types.SimpleNamespace(decode=lambda x: types.SimpleNamespace(
                sample=torch.nn.functional.interpolate(x[:, :3], scale_factor=8, mode="nearest")))
            engine.targets = np.load(target_file)
            engine.checkpoint_keys = {}
            plan = {"plan_hash": "unit-fixture", "classes": ["a", "b"],
                    "prepared_manifest": {"imagenet_labels": [0, 1]},
                    "protocol": {"device": "cpu", "cfg_scale": 4, "steps": 4, "guidance_steps": 2,
                                 "image_size": 256, "ipc": 3, "generation_batch_size": 2}}
            one = engine.generate(plan, root, 100, "no_mode", {"coefficients": [0, 0]})
            two = engine.generate(plan, root, 100, "cags", {"coefficients": [.03, .04]})
            self.assertEqual(one["noise_seeds"], two["noise_seeds"])
            self.assertEqual(len(two["images"]), 6)
            self.assertEqual(two["active_guidance_steps"], [2, 3])
            folder = root / "datasets/g100/cags"
            timestamp = (folder / "a/0000.png").stat().st_mtime_ns
            engine.generate(plan, root, 100, "cags", {"coefficients": [.03, .04]})
            self.assertEqual((folder / "a/0000.png").stat().st_mtime_ns, timestamp)
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                engine.generate(plan, root, 100, "cags", {"coefficients": [.04, .03]})
            Image.new("RGB", (10, 10)).save(folder / "extra.png")
            with self.assertRaisesRegex(ValueError, "Extra or missing"):
                verify_dataset(folder)
            engine.close()

    def test_small_feature_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            rng = np.random.default_rng(2)
            for cls, mean in [("a", 0), ("b", 3)]:
                np.save(folder / f"{cls}.npy", rng.normal(mean, 1, (12, 6)).astype(np.float32))
            profile = profile_from_features(folder, ["a", "b"], 3, 2, 2)
            self.assertEqual(profile["per_class"]["a"]["k"], 2)
            self.assertEqual(len(coefficient_vectors(profile)["cags"]), 2)

    def test_plan_selection_and_final_seed_separation(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            prep = root / "prepared"
            (prep / "profiles").mkdir(parents=True)
            stats = toy_stats()
            atomic_json(prep / "profiles/vae_auto.json", stats)
            split = {"root": str(root / "source"), "classes": stats["classes"], "seed": 1,
                     "train": {}, "tune": {c: [f"train/{c}/0.jpg"] for c in stats["classes"]}}
            for cls in stats["classes"]:
                source = root / "source/train" / cls
                source.mkdir(parents=True)
                Image.new("RGB", (64, 64)).save(source / "0.jpg")
                report = root / "report" / cls
                report.mkdir(parents=True)
                Image.new("RGB", (64, 64)).save(report / "1.jpg")
            atomic_json(prep / "split.json", split)
            manifest = {"classes": stats["classes"], "image_size": 256, "ipc": 10, "vae": "fixture",
                        "profiles": ["vae_auto"], "imagenet_labels": list(range(8)), "files": {
                            "split.json": file_hash(prep / "split.json"),
                            "profiles/vae_auto.json": file_hash(prep / "profiles/vae_auto.json")}}
            atomic_json(prep / "manifest.json", manifest)
            checkpoint = root / "fake.pt"
            checkpoint.write_bytes(b"test checkpoint; no generation performed")
            allocation_args = parser().parse_args([
                "plan", "--workdir", str(root), "--name", "allocation", "--suite", "allocation",
                "--checkpoint", str(checkpoint), "--report-dir", str(root / "report"), "--device", "cpu"])
            # This final-report plan must work before any tuning/selection exists.
            plan_experiment(allocation_args)
            allocation_dir = root / "runs/allocation"
            allocation = read_json_test(allocation_dir / "plan.json")
            self.assertIsNone(allocation["selection"])
            self.assertEqual(len(allocation["conditions"]), 6)
            self.assertEqual(len(list((allocation_dir / "queue").glob("gen_*.job.json"))), 30)
            eval_blocks = list((allocation_dir / "queue").glob("eval_*.job.json"))
            self.assertEqual(len(eval_blocks) * len(allocation["conditions"]), 90)
            self.assertEqual(allocation["protocol"]["checkpoint_rule"], "final")
            self.assertEqual(allocation["protocol"]["epochs"], 2000)
            self.assertEqual(allocation["protocol"]["evaluation_split"], "report")
            self.assertEqual(allocation["protocol"]["cfg_scale"], 4.)
            argv = ["plan", "--workdir", str(root), "--name", "tune", "--suite", "tune",
                    "--checkpoint", str(checkpoint), "--generation-seeds", "0,1,2",
                    "--training-seeds", "0", "--fixed-grid", "0.02,0.04,0.1", "--device", "cpu"]
            args = parser().parse_args(argv)
            plan_experiment(args)
            plan_dir = root / "runs/tune"
            plan = read_json_test(plan_dir / "plan.json")
            self.assertEqual(len(list((plan_dir / "queue").glob("gen_*.job.json"))), 51)
            for gen in [0, 1, 2]:
                for name in plan["conditions"]:
                    score = {"no_mode": 10, "cags": 14, "fixed_0p02": 13, "fixed_0p04": 12,
                             "fixed_0p1": 14.5, "cags_tune_entropy": 15}.get(name, 11)
                    path = plan_dir / "evaluation" / f"g{gen}/t0/convnet6/{name}.json"
                    atomic_json(path, {"plan_hash": plan["plan_hash"], "metrics": {"top1": score, "top5": 30},
                                       "selected_epoch": 2000, "hardware": {"gpu": "CPU"}})
            select_fixed(types.SimpleNamespace(plan=str(plan_dir), output=None))
            selection = plan_dir / "selection.json"
            self.assertEqual(read_json_test(selection)["best_fixed_by_arch"]["convnet6"]["coefficient"], .1)
            final_args = parser().parse_args(["plan", "--workdir", str(root), "--name", "final", "--suite", "full",
                                             "--checkpoint", str(checkpoint), "--selection", str(selection),
                                             "--generation-seeds", "100,101", "--report-dir", str(root / "report"),
                                             "--device", "cpu"])
            plan_experiment(final_args)
            final_plan = read_json_test(root / "runs/final/plan.json")
            self.assertEqual(len(final_plan["conditions"]), 20)
            selected_name = final_plan["selected_cags_references"]["convnet6"]
            self.assertEqual(selected_name, "cags_selected_convnet6")
            self.assertAlmostEqual(np.mean(final_plan["conditions"][selected_name]["coefficients"]),
                                   final_plan["conditions"]["selected_fixed_mean_convnet6"]["coefficients"][0])
            final_args.name = "invalid_overlap"
            final_args.generation_seeds = [0, 100]
            with self.assertRaisesRegex(ValueError, "disjoint"):
                plan_experiment(final_args)
            args.lr = .5
            with self.assertRaisesRegex(ValueError, "immutable"):
                plan_experiment(args)

    def test_cpu_training_reports_fixed_final_and_keeps_provenance(self):
        from rebuttal.evaluate import loader, train_one
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            classes = ["a", "b"]
            paths = []
            for i, cls in enumerate(classes):
                (root / "train" / cls).mkdir(parents=True)
                (root / "eval" / cls).mkdir(parents=True)
                for j in range(2):
                    array = np.full((64, 64, 3), 40 + i * 100 + j, dtype=np.uint8)
                    Image.fromarray(array).save(root / "train" / cls / f"{j}.png")
                    Image.fromarray(array).save(root / "eval" / cls / f"{j}.png")
                    paths.append((f"{cls}/{j}.png", i))
            cfg = {"device": "cpu", "deterministic": True, "threads": 1, "classifier_size": 64,
                   "ipc": 2, "lr": .1, "weight_decay": 1e-4, "epochs": 1, "precision": "fp32",
                   "batch_size": 4, "checkpoint_rule": "final", "eval_interval": 1, "save_checkpoints": True}
            out = root / "result.json"
            result = train_one(root / "train", classes, cfg, 0, "convnet6",
                               loader(root / "eval", paths, 64, 4, 0), None, out,
                               {"plan_hash": "fixture", "condition": "no_mode", "generation_seed": 100})
            self.assertEqual(result["metrics"]["n"], 4)
            self.assertEqual(result["selected_epoch"], 1)
            self.assertEqual(result["metrics"]["per_class_total"], [2, 2])
            self.assertTrue(out.with_suffix(".pt").exists())
            self.assertEqual(read_json_test(out)["plan_hash"], "fixture")

    def test_summary_does_not_pool_classifier_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = {"generation_seeds": [100, 101, 102], "training_seeds": [0, 1, 2],
                    "architectures": ["convnet6"], "conditions": {
                        "cags": {}, "no_mode": {}, "permuted_17": {"permutation_of": "cags"},
                        "cags_selected_convnet6": {"kind": "heldout_selected_cags"},
                        "selected_permuted_convnet6_17": {"permutation_of": "cags_selected_convnet6"}}}
            plan["plan_hash"] = digest(plan)
            atomic_json(root / "plan.json", plan)
            for index, gen in enumerate(plan["generation_seeds"]):
                for train in plan["training_seeds"]:
                    for name in plan["conditions"]:
                        path = root / "evaluation" / f"g{gen}/t{train}/convnet6/{name}.json"
                        score = {"no_mode": 20 + train, "cags": 20 + train + index + 1,
                                 "permuted_17": 20 + train + .5 * (index + 1),
                                 "cags_selected_convnet6": 25 + train + 2 * (index + 1),
                                 "selected_permuted_convnet6_17": 25 + train + index + 1}[name]
                        atomic_json(path, {"plan_hash": plan["plan_hash"], "metrics": {
                            "top1": score, "top5": 40},
                            "selected_epoch": 1, "hardware": {"gpu": "fixture"}})
            result = summarize(root)["by_architecture"]["convnet6"]["comparisons"]["cags_minus_no_mode"]
            self.assertEqual(result["n_generation"], 3)
            self.assertEqual(result["generation_differences_pp"], [1, 2, 3])
            self.assertAlmostEqual(result["mean_pp"], 2)
            self.assertIn("cags_minus_no_mode\t3\t2\t", (root / "summary.txt").read_text())
            self.assertIn("p_holm", (root / "summary.txt").read_text())
            self.assertTrue((root / "reviewer_evidence.csv").exists())
            self.assertEqual(read_json_test(root / 'runtime.json')['expected_fits'], 45)
            selected = summarize(root)["by_architecture"]["convnet6"]["comparisons"]
            self.assertEqual(selected["cags_selected_convnet6_minus_permuted_average"]["generation_differences_pp"], [1, 2, 3])
            path.unlink()
            with self.assertRaisesRegex(ValueError, "Missing"):
                summarize(root)
            partial = summarize(root, allow_partial=True)
            self.assertFalse(partial["complete"])
            self.assertEqual(partial["by_architecture"]["convnet6"]["comparisons"], {})
            self.assertIn("INCOMPLETE", (root / "summary.txt").read_text())
            from rebuttal.reporting import evidence_rows
            self.assertTrue(all(row['status'] == 'pending' for row in evidence_rows(partial, plan)))


def read_json_test(path):
    return json.loads(Path(path).read_text())


if __name__ == "__main__":
    unittest.main()
