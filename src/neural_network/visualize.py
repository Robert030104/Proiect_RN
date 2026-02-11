from pathlib import Path
import json
import numpy as np
import pandas as pd


def get_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def read_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def _safe_num_series(df, col):
    if col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce")
    if s.isna().all():
        return None
    return s


def plot_loss_curve(plt, hist, out_path):
    ep = _safe_num_series(hist, "epoch")
    tr = _safe_num_series(hist, "train_loss")
    va = _safe_num_series(hist, "val_loss")
    if ep is None or tr is None or va is None:
        return
    plt.figure()
    plt.plot(ep, tr, label="train_loss")
    plt.plot(ep, va, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve (Train vs Validation)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_metrics_evolution(plt, hist, out_path):
    ep = _safe_num_series(hist, "epoch")
    auc = _safe_num_series(hist, "val_auc")
    if ep is None or auc is None:
        return
    plt.figure()
    plt.plot(ep, auc, label="val_auc")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("Validation AUC Evolution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_learning_curves_final(plt, hist, out_path):
    ep = _safe_num_series(hist, "epoch")
    tr = _safe_num_series(hist, "train_loss")
    va = _safe_num_series(hist, "val_loss")
    auc = _safe_num_series(hist, "val_auc")
    if ep is None or tr is None or va is None or auc is None:
        return

    fig, ax1 = plt.subplots()
    ax1.plot(ep, tr, label="train_loss")
    ax1.plot(ep, va, label="val_loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    ax2 = ax1.twinx()
    ax2.plot(ep, auc, linestyle="--", label="val_auc")
    ax2.set_ylabel("AUC")

    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="best")

    plt.title("Final Learning Curves (Loss + AUC)")
    fig.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_bar(plt, df, metric_col, title, ylabel, out_path, top_n=12):
    if "run_id" not in df.columns or metric_col not in df.columns:
        return
    tmp = df.copy()
    tmp[metric_col] = pd.to_numeric(tmp[metric_col], errors="coerce")
    tmp = tmp.dropna(subset=[metric_col])
    if tmp.empty:
        return
    tmp = tmp.sort_values(metric_col, ascending=False).head(top_n)
    plt.figure()
    plt.bar(tmp["run_id"].astype(str), tmp[metric_col].astype(float))
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_pareto_f1_vs_fpr(plt, opt, out_path):
    if "val_f1" not in opt.columns or "val_fpr" not in opt.columns:
        return
    tmp = opt.copy()
    tmp["val_f1"] = pd.to_numeric(tmp["val_f1"], errors="coerce")
    tmp["val_fpr"] = pd.to_numeric(tmp["val_fpr"], errors="coerce")
    tmp = tmp.dropna(subset=["val_f1", "val_fpr"])
    if tmp.empty:
        return

    x = tmp["val_fpr"].astype(float).values
    y = tmp["val_f1"].astype(float).values

    order = np.argsort(x)
    x_s = x[order]
    y_s = y[order]

    pareto_x = []
    pareto_y = []
    best_y = -1.0
    for xi, yi in zip(x_s, y_s):
        if yi > best_y:
            pareto_x.append(xi)
            pareto_y.append(yi)
            best_y = yi

    plt.figure()
    plt.scatter(x, y)
    plt.plot(pareto_x, pareto_y, linestyle="--", label="Pareto frontier")
    plt.xlabel("Validation FPR (false alarms)")
    plt.ylabel("Validation F1")
    plt.title("Trade-off: False Alarms vs F1 (Pareto)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_metrics_summary_card(plt, test_json, final_json, out_path):
    lines = []
    if test_json:
        lines.append("TEST (baseline):")
        for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "auc"]:
            if k in test_json:
                lines.append(f"  {k}: {test_json[k]}")
        lines.append("")
    if final_json:
        lines.append("FINAL (optimized):")
        for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "auc", "threshold_used", "threshold_auto", "max_fpr"]:
            if k in final_json:
                lines.append(f"  {k}: {final_json[k]}")
        lines.append("")
    if not lines:
        return

    text = "\n".join(lines)
    plt.figure(figsize=(8, 4))
    plt.axis("off")
    plt.title("Metrics Summary", pad=10)
    plt.text(0.01, 0.95, text, va="top", ha="left", family="monospace")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    root = get_root()

    results_dir = root / "results"
    docs_results = root / "docs" / "results"
    docs_opt = root / "docs" / "optimization"

    ensure_dir(docs_results)
    ensure_dir(docs_opt)

    plt = try_import_matplotlib()
    if plt is None:
        print("matplotlib nu este instalat -> nu pot genera PNG-urile din docs/.")
        print("Instaleaza:  py -m pip install matplotlib")
        return

    hist_path = results_dir / "training_history.csv"
    opt_path = results_dir / "optimization_experiments.csv"
    test_path = results_dir / "test_metrics.json"
    final_path = results_dir / "final_metrics.json"

    if hist_path.exists():
        hist = pd.read_csv(hist_path)
        if "run_id" in hist.columns and len(hist) > 0:
            last_run = str(hist["run_id"].iloc[-1])
            hist = hist[hist["run_id"].astype(str) == last_run].copy()

        plot_loss_curve(plt, hist, docs_results / "loss_curve.png")
        plot_metrics_evolution(plt, hist, docs_results / "metrics_evolution.png")
        plot_learning_curves_final(plt, hist, docs_results / "learning_curves_final.png")

    if opt_path.exists():
        opt = pd.read_csv(opt_path)
        plot_bar(plt, opt, "val_accuracy", "Accuracy Comparison (Top Runs)", "Accuracy", docs_opt / "accuracy_comparison.png")
        plot_bar(plt, opt, "val_f1", "F1 Comparison (Top Runs)", "F1", docs_opt / "f1_comparison.png")

        if "best_auc" in opt.columns:
            plot_bar(plt, opt, "best_auc", "AUC Comparison (Top Runs)", "AUC", docs_opt / "auc_comparison.png")

        if "val_f1" in opt.columns and "val_fpr" in opt.columns:
            plot_pareto_f1_vs_fpr(plt, opt, docs_opt / "pareto_f1_vs_fpr.png")

    test_json = read_json(test_path)
    final_json = read_json(final_path)
    plot_metrics_summary_card(plt, test_json, final_json, docs_results / "metrics_summary.png")

    print("Done. PNG-urile au fost generate in docs/results si docs/optimization.")


if __name__ == "__main__":
    main()
