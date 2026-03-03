import argparse
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def load_metrics(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def panel_curve(ax, path, title):
    ax.set_title(title)
    if os.path.exists(path):
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.axis("off")
    else:
        ax.text(0.5, 0.5, f"Missing file:\n{path}", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])


def build_summary_lines(metrics):
    lines = [
        f"Generated UTC: {metrics.get('generated_utc', 'unknown')}",
        f"Commit: {metrics.get('git_commit', 'unknown')}",
    ]

    fixed = metrics.get("fixed_goal")
    if fixed:
        lines.extend(
            [
                "",
                "Fixed Goal:",
                f"- Episodes: {fixed['episodes']}",
                f"- Success rate: {fixed['success_rate']:.3f}",
                f"- Avg final distance: {fixed['avg_final_distance']:.3f}",
                f"- Avg return: {fixed['avg_return']:.3f}",
                f"- Avg episode length: {fixed['avg_episode_length']:.1f}",
            ]
        )

    arbitrary = metrics.get("arbitrary_goal")
    if arbitrary:
        lines.extend(
            [
                "",
                f"Arbitrary Goal ({arbitrary.get('goal_sampling', 'random')}):",
                f"- Episodes: {arbitrary['episodes']}",
                f"- Success rate: {arbitrary['success_rate']:.3f}",
                f"- Avg final distance: {arbitrary['avg_final_distance']:.3f}",
                f"- Avg return: {arbitrary['avg_return']:.3f}",
                f"- Avg episode length: {arbitrary['avg_episode_length']:.1f}",
            ]
        )
    return "\n".join(lines)


def chart_metrics(ax, metrics):
    labels = []
    success = []
    final_dist = []
    avg_return = []

    fixed = metrics.get("fixed_goal")
    if fixed:
        labels.append("Fixed")
        success.append(fixed["success_rate"])
        final_dist.append(fixed["avg_final_distance"])
        avg_return.append(fixed["avg_return"])

    arbitrary = metrics.get("arbitrary_goal")
    if arbitrary:
        labels.append("Arbitrary")
        success.append(arbitrary["success_rate"])
        final_dist.append(arbitrary["avg_final_distance"])
        avg_return.append(arbitrary["avg_return"])

    if not labels:
        ax.text(0.5, 0.5, "No metrics found", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    x = np.arange(len(labels))
    w = 0.25
    ax.bar(x - w, success, width=w, label="Success rate")
    ax.bar(x, final_dist, width=w, label="Avg final distance")
    ax.bar(x + w, avg_return, width=w, label="Avg return")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Key Metrics")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="best")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=str, default="results/latest_metrics.json")
    parser.add_argument("--fixed-curve", type=str, default="car_learning_curve.png")
    parser.add_argument("--arbitrary-curve", type=str, default="car_arbitrary_goal_curve.png")
    parser.add_argument("--out-image", type=str, default="results/report.png")
    parser.add_argument("--out-pdf", type=str, default="results/report.pdf")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_image), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_pdf), exist_ok=True)
    metrics = load_metrics(args.metrics)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("RL + MuJoCo Benchmark Report", fontsize=16, fontweight="bold")

    axes[0, 0].axis("off")
    axes[0, 0].text(
        0.01,
        0.99,
        build_summary_lines(metrics),
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
    )
    chart_metrics(axes[0, 1], metrics)
    panel_curve(axes[1, 0], args.fixed_curve, "Fixed Goal Learning Curve")
    panel_curve(axes[1, 1], args.arbitrary_curve, "Arbitrary Goal Learning Curve")

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(args.out_image, dpi=220)
    fig.savefig(args.out_pdf)
    plt.close(fig)

    print(f"Saved report image: {args.out_image}")
    print(f"Saved report pdf: {args.out_pdf}")


if __name__ == "__main__":
    main()
