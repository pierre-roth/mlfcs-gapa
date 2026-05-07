from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect piroth2 LOB pretraining comparison summaries.")
    parser.add_argument("--root", default="/cluster/project/math/piroth/mlfcs-gapa/artifacts_piroth2")
    parser.add_argument("--stamp", default="", help="Optional run stamp, for example 20260505_143142.")
    parser.add_argument("--format", choices=["markdown", "csv"], default="markdown")
    args = parser.parse_args()

    rows = []
    patterns = (
        [f"piroth2_pretraincmp_*_{args.stamp}", f"piroth2_pretrainthr_*_{args.stamp}"]
        if args.stamp
        else ["piroth2_pretraincmp_*", "piroth2_pretrainthr_*"]
    )
    run_dirs = []
    for pattern in patterns:
        run_dirs.extend(Path(args.root).glob(pattern))
    for run_dir in sorted(set(run_dirs)):
        summary_files = sorted((run_dir / "models").glob("*_pretrain_summary.json"))
        for summary_file in summary_files:
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
            parts = run_dir.name.split("_")
            if len(parts) < 6:
                continue
            dataset = parts[2]
            symbol = parts[3]
            model_type = summary.get("model_type", parts[4])
            final = summary.get("final", {})
            rows.append(
                {
                    "dataset": dataset,
                    "symbol": symbol,
                    "model": model_type,
                    "parameters": summary.get("parameters", ""),
                    "train_accuracy": final.get("train_accuracy", ""),
                    "eval_accuracy": final.get("eval_accuracy", ""),
                    "eval_f1_macro": final.get("eval_f1_macro", ""),
                    "eval_loss": final.get("eval_loss", ""),
                    "train_samples": final.get("train_samples", ""),
                    "eval_samples": final.get("eval_samples", ""),
                    "train_label_counts": summary.get("train_label_counts", ""),
                    "eval_label_counts": summary.get("eval_label_counts", ""),
                    "class_weight_mode": summary.get("class_weight_mode", ""),
                    "run_name": run_dir.name,
                }
            )
    rows.sort(key=lambda row: (row["dataset"], row["symbol"], row["model"]))

    headers = [
        "dataset",
        "symbol",
        "model",
        "parameters",
        "train_accuracy",
        "eval_accuracy",
        "eval_f1_macro",
        "eval_loss",
        "train_samples",
        "eval_samples",
        "train_label_counts",
        "eval_label_counts",
        "class_weight_mode",
        "run_name",
    ]
    if args.format == "csv":
        print(",".join(headers))
        for row in rows:
            print(",".join(_format_value(row[header]) for header in headers))
        return

    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        print("| " + " | ".join(_format_value(row[header]) for header in headers) + " |")


def _format_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
