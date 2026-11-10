"""Evidence mapping and measured runtime estimates, separate from historical tables."""
import csv
from pathlib import Path
from statistics import median

from .common import atomic_json, read_json


def runtime_estimate(folder, plan, workers=4):
    folder = Path(folder)
    generation, training = [], []
    for seed in plan['generation_seeds']:
        for condition in plan['conditions']:
            path = folder / 'datasets' / f'g{seed}' / condition / 'manifest.json'
            if path.exists():
                value = read_json(path)
                if value.get('plan_hash') == plan['plan_hash'] and value.get('elapsed_s', 0) > 0:
                    generation.append(value['elapsed_s'])
        for train in plan['training_seeds']:
            for arch in plan['architectures']:
                for condition, spec in plan['conditions'].items():
                    if spec.get('only_arch', arch) != arch:
                        continue
                    path = folder / 'evaluation' / f'g{seed}' / f't{train}' / arch / f'{condition}.json'
                    if path.exists():
                        value = read_json(path)
                        if (value.get('plan_hash') == plan['plan_hash'] and 'metrics' in value
                                and value.get('elapsed_s', 0) > 0):
                            training.append(value['elapsed_s'])
    ng = len(plan['generation_seeds']) * len(plan['conditions'])
    nt = len(plan['generation_seeds']) * len(plan['training_seeds']) * sum(
        c.get('only_arch', arch) == arch for arch in plan['architectures'] for c in plan['conditions'].values())
    remaining = None
    if generation and training:
        remaining = (max(0, ng - len(generation)) * median(generation)
                     + max(0, nt - len(training)) * median(training)) / 3600
    return {'plan_hash': plan['plan_hash'], 'measured_datasets': len(generation), 'expected_datasets': ng,
            'measured_fits': len(training), 'expected_fits': nt,
            'median_generation_seconds': median(generation) if generation else None,
            'median_training_seconds': median(training) if training else None,
            'estimated_remaining_gpu_hours': remaining,
            'ideal_remaining_wall_hours': remaining / workers if remaining is not None else None,
            'workers_assumed': workers,
            'limitations': 'Estimate for this frozen plan only; excludes scheduler wait, startup, failures, and idle capacity. Not a completion deadline.'}


def evidence_rows(summary, plan):
    rows = []
    for arch in plan['architectures']:
        comparisons = summary['by_architecture'][arch]['comparisons']
        targets = [('7hx5 W1.1; 76Xw W2', '类别分配与平均强度', 'cags_minus_fixed_mean'),
                   ('7hx5 W1.1', '类别分配与随机置换', 'cags_minus_permuted_average'),
                   ('7hx5 W1.2; PrL8 W1', '同协议下相对关闭模态引导的增益', 'cags_minus_no_mode')]
        if plan.get('suite') == 'full':
            reference = plan.get('selected_cags_references', {}).get(arch, 'cags')
            targets += [('7hx5 Q2; 76Xw W2', '双方留出集选参后的对比', f'{reference}_minus_fixed_best_{arch}')]
            targets += [('PrL8 W2; 76Xw Q3', '统一因素消融', 'cags_minus_' + name)
                        for name in ['entropy', 'dispersion', 'four_factor', 'entropy_mean_matched',
                                     'dispersion_mean_matched', 'four_factor_mean_matched']]
        for reviewer, question, key in targets:
            comparison = comparisons.get(key)
            row = {'reviewer': reviewer, 'question': question, 'architecture': arch, 'comparison': key,
                   'status': 'pending', 'n_generation': None, 'difference_pp': None,
                   'ci95_low': None, 'ci95_high': None, 'p_holm': None}
            if plan.get('suite') == 'tune':
                row['status'] = 'tuning_only_not_final_evidence'
            elif summary['complete'] and comparison is not None:
                low, high = comparison['ci95_pp'] or (None, None)
                p = comparison.get('p_holm')
                status = 'inconclusive'
                if p is not None and p < .05 and low is not None:
                    status = 'positive_difference' if low > 0 else 'negative_difference' if high < 0 else status
                row.update(status=status, n_generation=comparison['n_generation'],
                           difference_pp=comparison['mean_pp'], ci95_low=low, ci95_high=high, p_holm=p)
            rows.append(row)
    return rows


def write_reports(folder, summary, plan):
    folder = Path(folder)
    rows = evidence_rows(summary, plan)
    if rows:
        with (folder / 'reviewer_evidence.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    atomic_json(folder / 'runtime.json', runtime_estimate(folder, plan))
