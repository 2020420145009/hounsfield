#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

TASK_SUBTASKS = {
    "task1": ["Anatomical_observation", "Pathological_observation"],
    "task2": ["Abnormality_feature", "Abnormality_position", "Abnormality_type", "Diagnosis"],
    "task3": ["Diameter", "Size", "Thickness"],
}
JUDGE_METRICS = ["judge_pass_rate", "judge_score_mean"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LLM-judge metric comparison for open tasks.")
    parser.add_argument("--task", choices=sorted(TASK_SUBTASKS.keys()), required=True)
    parser.add_argument(
        "--model-result",
        action="append",
        nargs=2,
        metavar=("MODEL_NAME", "RESULTS_ROOT"),
        required=True,
        help="Pair of model name and results root. Repeat for each model.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def compute_subtask_metrics(result_csv: Path):
    if not result_csv.exists() or result_csv.stat().st_size == 0:
        return None
    rows = list(csv.DictReader(open(result_csv, newline="")))
    if not rows:
        return None

    judged = 0
    pass_sum = 0
    score_sum = 0.0
    for row in rows:
        try:
            judge_pass = int(str(row.get("judge_pass", "")).strip())
            judge_score = float(str(row.get("judge_score", "")).strip())
        except ValueError:
            continue
        judged += 1
        pass_sum += judge_pass
        score_sum += judge_score

    if judged == 0:
        return {"judge_pass_rate": None, "judge_score_mean": None, "judged": 0, "total": len(rows)}
    return {
        "judge_pass_rate": pass_sum / judged,
        "judge_score_mean": score_sum / judged,
        "judged": judged,
        "total": len(rows),
    }


def main() -> None:
    args = parse_args()
    task = args.task
    subtasks = TASK_SUBTASKS[task]
    model_roots = {name: Path(root).resolve() for name, root in args.model_result}
    model_names = list(model_roots.keys())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subtask", "metric"] + model_names)

        macro_values = {metric: {name: [] for name in model_names} for metric in JUDGE_METRICS}
        for subtask in subtasks:
            for metric in JUDGE_METRICS:
                row = [subtask, metric]
                for model_name in model_names:
                    result_csv = model_roots[model_name] / task / subtask / "eval_open_vqa_judge.csv"
                    metrics = compute_subtask_metrics(result_csv)
                    if metrics is None or metrics[metric] is None:
                        row.append("missing")
                    else:
                        macro_values[metric][model_name].append(metrics[metric])
                        row.append(f"{metrics[metric]:.6f}")
                writer.writerow(row)

            row = [subtask, "judged_over_total"]
            for model_name in model_names:
                result_csv = model_roots[model_name] / task / subtask / "eval_open_vqa_judge.csv"
                metrics = compute_subtask_metrics(result_csv)
                if metrics is None:
                    row.append("missing")
                else:
                    row.append(f"{metrics['judged']}/{metrics['total']}")
            writer.writerow(row)

        for metric in JUDGE_METRICS:
            macro_row = [f"{task.upper()}_MACRO", metric]
            for model_name in model_names:
                vals = macro_values[metric][model_name]
                macro_row.append(f"{(sum(vals) / len(vals)):.6f}" if vals else "missing")
            writer.writerow(macro_row)


if __name__ == "__main__":
    main()
