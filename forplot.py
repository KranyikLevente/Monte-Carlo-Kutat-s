import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_runs_line_scatter(csv_path="simulation_results/summary_metrics.csv", output_dir="simulation_results"):
    df = pd.read_csv(csv_path)

    methods = [
        ("Surface", "Surface Tracking", "#1f77b4"),           # Blue
        ("Woodcock", "Woodcock Tracking (Delta)", "#ff7f0e"),  # Orange
        ("Hybrid_Woodcock", "Hybrid Woodcock", "#2ca02c"),     # Green
        ("Hybrid_Surface", "Hybrid Surface Tracking", "#d62728") # Red
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Results of Monte Carlo Tracking Methods per Run (1–10)", fontsize=15, fontweight="bold")

    for method_id, label, color in methods:
        sub_df = df[df["Method_ID"] == method_id].sort_values("Run")
        runs = sub_df["Run"]
        mfp = sub_df["Mean_Free_Path"]
        runtime = sub_df["Runtime_s"]

        ax1.plot(runs, mfp, color=color, linestyle="-", linewidth=1.8, alpha=0.8, label=label)
        ax1.scatter(runs, mfp, color=color, edgecolors="black", s=60, zorder=3)
        ax1.axhline(mfp.mean(), color=color, linestyle="--", linewidth=1.0, alpha=0.5)

        ax2.plot(runs, runtime, color=color, linestyle="-", linewidth=1.8, alpha=0.8, label=label)
        ax2.scatter(runs, runtime, color=color, edgecolors="black", s=60, zorder=3)
        ax2.axhline(runtime.mean(), color=color, linestyle="--", linewidth=1.0, alpha=0.5)

    ax1.set_title("Measured Mean Free Path over Runs", fontsize=13)
    ax1.set_xlabel("Run Number", fontsize=11)
    ax1.set_ylabel("Mean Free Path", fontsize=11)
    ax1.set_xticks(range(1, 11))
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="best", framealpha=0.9)

    ax2.set_title("Runtime over Runs", fontsize=13)
    ax2.set_xlabel("Run Number", fontsize=11)
    ax2.set_ylabel("Runtime (seconds)", fontsize=11)
    ax2.set_xticks(range(1, 11))
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="best", framealpha=0.9)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "mfp_and_runtime_per_run.png")
    plt.savefig(out_file, dpi=300)
    print(f"Figure saved: {out_file}")
    plt.show()

if __name__ == "__main__":
    plot_runs_line_scatter()