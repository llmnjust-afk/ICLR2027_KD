"""Generation-level paired statistics. Training seeds are never pooled as independent generations."""
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from .common import atomic_json, digest, read_json, read_plan


def paired_stats(differences):
    values = np.asarray(differences, dtype=float)
    mean = float(values.mean())
    n = len(values)
    sd = float(values.std(ddof=1)) if n > 1 else None
    if n < 2:
        return {"n_generation": n, "mean_pp": mean, "sd_pp": sd, "ci95_pp": None,
                "p_two_sided": None, "cohen_dz": None, "generation_differences_pp": values.tolist()}
    margin = float(stats.t.ppf(.975, n - 1) * sd / math.sqrt(n))
    p = float(stats.ttest_1samp(values, 0).pvalue) if sd > 0 else None
    return {"n_generation": n, "mean_pp": mean, "sd_pp": sd,
            "ci95_pp": [mean - margin, mean + margin], "p_two_sided": p,
            "cohen_dz": mean / sd if sd > 0 else None,
            "generation_differences_pp": values.tolist()}


def holm(comparisons):
    keys = sorted((key for key in comparisons if comparisons[key]["p_two_sided"] is not None),
                  key=lambda key: comparisons[key]["p_two_sided"])
    running = 0
    for index, key in enumerate(keys):
        running = max(running, min(1, comparisons[key]["p_two_sided"] * (len(keys) - index)))
        comparisons[key]["p_holm"] = running


def summary_text(summary, plan):
    """A separate result table; never merge historical best-epoch numbers."""
    protocol = plan.get("protocol", {})
    lines = ["CAGS paired allocation / rebuttal results",
             f"Plan: {summary['plan_hash']}",
             f"Complete: {summary['complete']}; missing fits: {len(summary['missing'])}",
             f"Checkpoint rule: {protocol.get('checkpoint_rule', 'not recorded')}; "
             f"epochs: {protocol.get('epochs', 'not recorded')}",
             "Unit: independent generation replicate, averaging paired classifier seeds.",
             "SD is between-generation sample SD; CI is a paired 95% t interval.",
             summary["scope"], ""]

    def number(value):
        return "NA" if value is None else f"{value:.6g}"

    for arch, results in summary["by_architecture"].items():
        lines += [arch, "condition\tn_generation\tmean_top1_pct\tsd_between_generation"]
        for name, row in results["conditions"].items():
            lines.append(f"{name}\t{len(row['generation_means'])}\t{number(row['mean_top1'])}"
                         f"\t{number(row['sd_between_generation'])}")
        lines.append("comparison\tn_generation\tdifference_pp\tci95_low\tci95_high\tp_two_sided\tp_holm")
        for name, row in results["comparisons"].items():
            low, high = row["ci95_pp"] or (None, None)
            lines.append(f"{name}\t{row['n_generation']}\t{number(row['mean_pp'])}"
                         f"\t{number(low)}\t{number(high)}\t{number(row['p_two_sided'])}"
                         f"\t{number(row.get('p_holm'))}")
        lines.append("")
    if not summary["complete"]:
        lines.append("INCOMPLETE: descriptive progress only; no inferential comparisons.")
    return "\n".join(lines) + "\n"


def summarize(plan_dir, allow_partial=False):
    plan_dir = Path(plan_dir)
    plan = read_plan(plan_dir / "plan.json")
    rows, missing = [], []
    for gen_seed in plan["generation_seeds"]:
        for train_seed in plan["training_seeds"]:
            for arch in plan["architectures"]:
                for name, condition in plan["conditions"].items():
                    if condition.get("only_arch", arch) != arch:
                        continue
                    path = plan_dir / "evaluation" / f"g{gen_seed}" / f"t{train_seed}" / arch / f"{name}.json"
                    if not path.exists():
                        missing.append(str(path.relative_to(plan_dir)))
                        continue
                    result = read_json(path)
                    if result["plan_hash"] != plan["plan_hash"]:
                        raise ValueError(f"Stale or mixed evaluation: {path}")
                    rows.append({"architecture": arch, "condition": name, "generation_seed": gen_seed,
                                 "training_seed": train_seed, "top1": result["metrics"]["top1"],
                                 "top5": result["metrics"]["top5"], "selected_epoch": result["selected_epoch"],
                                 "gpu": result["hardware"]["gpu"]})
    if missing and not allow_partial:
        raise ValueError(f"Missing {len(missing)} evaluations. Use status, or --allow-partial for descriptive progress only.")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["architecture"], row["condition"], row["generation_seed"])].append(row)
    output = {"plan_hash": plan["plan_hash"], "complete": not missing, "missing": missing,
              "unit": "independent generation replicate, after averaging its paired classifier seeds",
              "scope": "conditional on frozen preparation cache, data split, weights, coefficient permutations, and evaluation protocol",
              "by_architecture": {}}
    for arch in plan["architectures"]:
        summaries = {}
        for name, condition in plan["conditions"].items():
            if condition.get("only_arch", arch) != arch:
                continue
            values = []
            for gen_seed in plan["generation_seeds"]:
                records = grouped[(arch, name, gen_seed)]
                if {r["training_seed"] for r in records} != set(plan["training_seeds"]):
                    continue
                values.append({"generation_seed": gen_seed,
                               "top1": float(np.mean([r["top1"] for r in records])),
                               "sd_training": float(np.std([r["top1"] for r in records], ddof=1)) if len(records) > 1 else None})
            if values:
                summaries[name] = {"mean_top1": float(np.mean([v["top1"] for v in values])),
                                   "sd_between_generation": float(np.std([v["top1"] for v in values], ddof=1)) if len(values) > 1 else None,
                                   "generation_means": values}
        comparisons = {}
        if not missing and "cags" in summaries:
            references = [name for name in summaries if name == "cags" or
                          plan["conditions"][name].get("kind") == "heldout_selected_cags"]
            for reference in references:
                cags = {v["generation_seed"]: v["top1"] for v in summaries[reference]["generation_means"]}
                for name, summary in summaries.items():
                    if name == reference:
                        continue
                    comparisons[f"{reference}_minus_{name}"] = paired_stats([
                        cags[v["generation_seed"]] - v["top1"] for v in summary["generation_means"]])
                permuted = [name for name in summaries if plan["conditions"][name].get("permutation_of") == reference]
                if permuted:
                    perm_maps = [{v["generation_seed"]: v["top1"] for v in summaries[name]["generation_means"]}
                                 for name in permuted]
                    comparisons[f"{reference}_minus_permuted_average"] = paired_stats([
                        cags[seed] - float(np.mean([values[seed] for values in perm_maps]))
                        for seed in plan["generation_seeds"]])
            holm(comparisons)
        output["by_architecture"][arch] = {"conditions": summaries, "comparisons": comparisons}
    atomic_json(plan_dir / "summary.json", output)
    with (plan_dir / "raw_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["architecture", "condition", "generation_seed", "training_seed", "top1", "top5", "selected_epoch", "gpu"])
        writer.writeheader()
        writer.writerows(rows)
    (plan_dir / "summary.txt").write_text(summary_text(output, plan), encoding="utf-8")
    from .reporting import write_reports
    write_reports(plan_dir, output, plan)
    return output
