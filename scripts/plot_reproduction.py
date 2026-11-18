"""Export curves from completed shared records without changing original files."""

import argparse
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "research-workspace/src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/matplotlib"))

from research_workspace import load_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_directory", type=Path)
    parser.add_argument(
        "--output-directory", type=Path,
        help="New output directory (relative to the working directory); defaults to BATCH/curves",
    )
    args = parser.parse_args()
    batch = args.batch_directory.resolve()
    if not batch.is_relative_to(ROOT / "results"):
        parser.error("Use a formal batch under results/")
    summary = json.loads((batch / "summary.json").read_text())
    records = []
    for row in summary["seeds"]:
        run = load_run(row["attempt"], require_ended=True)
        if run["status"]["state"] != "success":
            raise ValueError("Plot completed successful attempts only")
        records.append((row["seed"], run["metrics"]))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = args.output_directory.resolve() if args.output_directory else batch / "curves"
    output.mkdir(exist_ok=False)
    fields = ["seed", "stage", "epoch", "train_mse", "test_mse", "train_loss",
              "test_loss", "train_accuracy", "test_accuracy"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), layout="constrained")
    colors = plt.get_cmap("tab10").colors
    evaluation_lines = [[], []]
    training_lines = [[], []]
    with (output / "curves.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, (seed, rows) in enumerate(records):
            ga = [row["metrics"] for row in rows if "ga/epoch" in row["metrics"]]
            dft = [row["metrics"] for row in rows if "dft/epoch" in row["metrics"]]
            color = colors[index]
            for stage, values in (("ga", ga), ("dft", dft)):
                for metrics in values:
                    record = {"seed": seed, "stage": stage,
                              **{key.removeprefix(stage + "/"): value
                                 for key, value in metrics.items()
                                 if key.removeprefix(stage + "/") in fields}}
                    writer.writerow(record)
            evaluation_lines[0].extend(axes[0].plot(
                [v["ga/epoch"] for v in ga], [v["ga/test_mse"] for v in ga],
                color=color, label=f"seed {seed} · Validation MSE"))
            training_lines[0].extend(axes[0].plot(
                [v["ga/epoch"] for v in ga], [v["ga/train_mse"] for v in ga],
                color=color, linestyle="--", alpha=.65, label=f"seed {seed} · Train MSE"))
            evaluation_lines[1].extend(axes[1].plot(
                [v["dft/epoch"] for v in dft], [100*v["dft/test_accuracy"] for v in dft],
                color=color, marker="o", markersize=3, label=f"seed {seed} · Test accuracy"))
            training_lines[1].extend(axes[1].plot(
                [v["dft/epoch"] for v in dft], [100*v["dft/train_accuracy"] for v in dft],
                color=color, linestyle="--", alpha=.65,
                label=f"seed {seed} · Train accuracy"))
    axes[0].set(title="GA: CMC ImageNet100", xlabel="GA epoch", ylabel="Reconstruction MSE")
    axes[0].set_yscale("log")
    axes[1].set(title="DFT: CIFAR-100", xlabel="DFT epoch", ylabel="Accuracy (%)")
    for index, ax in enumerate(axes):
        ax.grid(alpha=.25)
        # Matplotlib fills legend columns first, pairing evaluation/training by seed in each row.
        ax.legend(handles=evaluation_lines[index] + training_lines[index],
                  loc="upper center", bbox_to_anchor=(.5, -.19), ncol=2,
                  fontsize=9, handlelength=3, columnspacing=1.5)
    fig.suptitle("AE-PSL / ViT-B/32 / split 5 / latent 64\nTraining progress by epoch")
    fig.supxlabel(
        "Solid: validation/test; dashed: training; colors: seeds 42, 43, 44\n"
        "GA evaluation: CMC ImageNet100 validation set · DFT evaluation: CIFAR-100 test set",
        fontsize=9,
    )
    for extension in ("png", "pdf"):
        fig.savefig(output / f"training-curves.{extension}", dpi=180)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
