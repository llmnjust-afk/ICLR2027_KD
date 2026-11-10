"""Read-only, replication-aware per-class analyses of existing result files."""
from pathlib import Path
import csv
import numpy as np
from scipy.stats import pearsonr, spearmanr
from rebuttal.common import atomic_json, read_json, read_plan, file_hash
from rebuttal.statistics import paired_stats


def correlations(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {'pearson_r': None, 'spearman_rho': None}
    return {'pearson_r': float(pearsonr(x, y).statistic), 'spearman_rho': float(spearmanr(x, y).statistic)}


def analyze(folder, output):
    folder, output = Path(folder), Path(output)
    plan = read_plan(folder / 'plan.json')
    profile = Path(plan['protocol']['prepared']) / 'profiles' / (plan['profile'] + '.json')
    expected = plan['prepared_manifest']['files']['profiles/' + plan['profile'] + '.json']
    if file_hash(profile) != expected:
        raise ValueError('Complexity profile differs from experiment provenance.')
    stats = read_json(profile)
    classes = plan['classes']
    records = [stats['per_class'][c] for c in classes]
    factors = {k: np.array([r[k] for r in records]) for k in ['entropy_normalized', 'dispersion', 'separability']}
    dispersion = factors['dispersion']
    factors['complexity_H05_V05'] = .5 * factors['entropy_normalized'] + .5 * (dispersion - dispersion.min()) / max(np.ptp(dispersion), 1e-8)
    animal_set = set((Path(__file__).parent / 'code/misc/class_animals_subset.txt').read_text().split())
    groups = {'animals': np.array([c in animal_set for c in classes]),
              'other_classes': np.array([c not in animal_set for c in classes])}
    rows, results, missing, provenance = [], {}, [], {}
    for arch in plan['architectures']:
        results[arch] = {}
        for name, spec in plan['conditions'].items():
            if spec.get('only_arch', arch) != arch:
                continue
            by_generation = {}
            for g in plan['generation_seeds']:
                values = []
                models = set()
                for t in plan['training_seeds']:
                    path = folder / 'evaluation' / f'g{g}' / f't{t}' / arch / (name + '.json')
                    if not path.exists():
                        missing.append(str(path))
                        continue
                    result = read_json(path)
                    if result['plan_hash'] != plan['plan_hash'] or result['condition'] != name or result['train_seed'] != t:
                        raise ValueError('Mixed result identity: ' + str(path))
                    metric = result['metrics']
                    counts = np.asarray(metric['per_class_total'])
                    correct = np.asarray(metric['per_class_correct'])
                    if counts.shape != (len(classes),) or np.any(counts <= 0) or np.any(correct < 0) or np.any(correct > counts):
                        raise ValueError('Invalid per-class counts: ' + str(path))
                    values.append(100 * correct / counts)
                    models.add(result['hardware']['gpu'])
                    provenance[str(path)] = file_hash(path)
                if len(values) == len(plan['training_seeds']):
                    accuracy = np.mean(values, axis=0)
                    by_generation[g] = {'accuracy': accuracy, 'gpu_models': sorted(models)}
                    for i, cls in enumerate(classes):
                        rows.append({'architecture': arch, 'condition': name, 'generation_seed': g,
                                     'class': cls, 'group': 'animals' if cls in animal_set else 'other_classes',
                                     'top1_pct': float(accuracy[i]), **{k: float(v[i]) for k, v in factors.items()}})
            results[arch][name] = by_generation
    report = {'source_run': str(folder), 'plan_hash': plan['plan_hash'], 'complete': not missing,
              'missing_fit_count': len(missing), 'source_result_sha256': provenance,
              'unit': 'Training-seed average within each generation; correlations over classes reported separately by generation.',
              'scope': 'Descriptive post-hoc correlations, not proof of optimal guidance or causal effects. Partial runs have no inferential intervals.',
              'groups': {k: [c for c, include in zip(classes, mask) if include] for k, mask in groups.items()},
              'by_architecture': {}}
    for arch, conditions in results.items():
        comparisons = {}
        if 'cags' not in conditions:
            continue
        for baseline in ['no_mode', 'fixed_mean', 'fixed_best_convnet6']:
            if baseline not in conditions:
                continue
            generations, group_diffs = [], {k: [] for k in groups}
            for g in plan['generation_seeds']:
                if g not in conditions['cags'] or g not in conditions[baseline]:
                    continue
                a, b = conditions['cags'][g], conditions[baseline][g]
                if a['gpu_models'] != b['gpu_models']:
                    raise ValueError('Hardware mismatch in paired class analysis.')
                gain = a['accuracy'] - b['accuracy']
                generations.append({'generation_seed': g, 'factors_vs_gain': {k: correlations(v, gain) for k, v in factors.items()},
                                    'factors_vs_baseline_accuracy': {k: correlations(v, b['accuracy']) for k, v in factors.items()}})
                for group, mask in groups.items():
                    if mask.any():
                        group_diffs[group].append(float(gain[mask].mean()))
            complete_pair = len(generations) == len(plan['generation_seeds'])
            comparisons['cags_minus_' + baseline] = {'generation_correlations': generations,
                'group_gain_generation_values_pp': group_diffs,
                'group_paired_intervals': {k: paired_stats(v) for k, v in group_diffs.items() if v} if complete_pair and not missing else None}
        # The already completed held-out fixed grid permits class response analysis
        # without new GPU training. Select a coefficient leaving one generation out.
        fixed = sorted((0.0 if spec.get('kind') == 'no_mode' else float(spec['value']), name)
                       for name, spec in plan['conditions'].items()
                       if (spec.get('kind') == 'no_mode' or spec.get('kind') == 'fixed' and 'value' in spec)
                       and name in conditions)
        response = []
        if plan['suite'] == 'tune' and len(fixed) >= 3 and not missing:
            for heldout in plan['generation_seeds']:
                training = [g for g in plan['generation_seeds'] if g != heldout]
                scores = np.array([np.mean([conditions[name][g]['accuracy'] for g in training], axis=0) for _, name in fixed])
                chosen = scores.argmax(axis=0)  # Smaller coefficient breaks ties.
                coefficients = np.array([v for v, _ in fixed])[chosen]
                evaluated = np.array([conditions[name][heldout]['accuracy'] for _, name in fixed])
                response.append({'heldout_generation': heldout, 'selected_coefficients_by_class': coefficients.tolist(),
                    'heldout_per_class_top1': evaluated[chosen, np.arange(len(classes))].tolist(),
                    'factor_vs_selected_coefficient': {k: correlations(v, coefficients) for k, v in factors.items()},
                    'note': 'Generation-level cross-validation on the tuning split, not final validation accuracy; discrete noisy optima, no monotonicity guarantee.'})
        report['by_architecture'][arch] = {'paired_class_analyses': comparisons, 'fixed_grid_leave_one_generation_out': response}
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / 'per_class_analysis.json', report)
    if rows:
        with (output / 'per_class.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f'Per-class analysis: {folder}; missing {len(missing)} fits; output {output}', flush=True)
    return report
