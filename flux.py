import csv
import os
import numpy as np


def evaluate_runs(run_indices, output_dir="simulation_results"):
    run_indices = list(run_indices)
    if not run_indices:
        print("Error: No run indices provided.")
        return

    method_keys = [
        ("Surface_flux", "Surface Tracking"),
        ("Woodcock_flux", "Woodcock Tracking (Delta)"),
        ("Hybrid_Woodcock_flux", "Hybrid Woodcock Tracking"),
        ("Hybrid_Surface_flux", "Hybrid Surface Tracking"),
    ]

    loaded_runs = []
    missing_runs = []

    for idx in run_indices:
        npz_path = os.path.join(output_dir, f"results_run_{idx:02d}.npz")
        if os.path.exists(npz_path):
            loaded_runs.append((idx, np.load(npz_path)))
        else:
            missing_runs.append(idx)

    if not loaded_runs:
        print("Error: No .npz files could be loaded.")
        return

    valid_runs = [idx for idx, _ in loaded_runs]
    valid_run_set = set(valid_runs)
    run_range_str = f"{min(valid_runs):02d}_to_{max(valid_runs):02d}"

    log_file_path = os.path.join(output_dir, f"evaluation_logs_runs_{run_range_str}.txt")

    with open(log_file_path, "w", encoding="utf-8") as log_file:
        def log_and_print(text):
            print(text)
            log_file.write(text + "\n")

        if missing_runs:
            log_and_print(f"Warning: Missing .npz files for runs: {missing_runs}")

        csv_path = os.path.join(output_dir, "summary_metrics.csv")
        runtimes_by_method = {name: [] for _, name in method_keys}
        mfp_by_method = {name: [] for _, name in method_keys}

        if os.path.exists(csv_path):
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    r_idx = int(row["Run"])
                    m_name = row["Method_Name"]
                    if r_idx in valid_run_set and m_name in runtimes_by_method:
                        runtimes_by_method[m_name].append(float(row["Runtime_s"]))
                        mfp_by_method[m_name].append(float(row["Mean_Free_Path"]))

        log_and_print(f"=== EVALUATION ACROSS {len(loaded_runs)} RUNS: {valid_runs} ===")
        log_and_print(f"Directory: {output_dir}\n")

        eval_summary = {}
        out_stats = {}

        for key, display_name in method_keys:
            flux_stack = [data[key] for _, data in loaded_runs if key in data]

            if not flux_stack:
                log_and_print(f"--- {display_name}: '{key}' not found in loaded files ---")
                continue

            arr = np.array(flux_stack, dtype=float)

            mean_flux = np.mean(arr, axis=0)
            mean_sq_flux = np.mean(arr ** 2, axis=0)
            std_flux = np.sqrt(np.maximum(mean_sq_flux - mean_flux ** 2, 0.0))

            short = key.replace("_flux", "")
            out_stats[f"{short}_flux_mean"] = mean_flux
            out_stats[f"{short}_flux_std"] = std_flux

            global_mean_flux = float(np.mean(mean_flux))
            global_std_flux = float(np.mean(std_flux))
            max_voxel_mean = float(np.max(mean_flux))
            max_voxel_std = float(np.max(std_flux))

            nonzero_mask = mean_flux > 0
            if np.any(nonzero_mask):
                rel_unc_map = np.zeros_like(mean_flux)
                rel_unc_map[nonzero_mask] = std_flux[nonzero_mask] / mean_flux[nonzero_mask]
                mean_rel_unc = float(np.mean(rel_unc_map[nonzero_mask]))
            else:
                mean_rel_unc = np.nan

            rt_list = runtimes_by_method[display_name]
            if rt_list:
                rt_arr = np.array(rt_list, dtype=float)
                mean_rt = float(np.mean(rt_arr))
                std_rt = float(np.sqrt(np.mean(rt_arr ** 2) - np.mean(rt_arr) ** 2))
                total_rt = float(np.sum(rt_arr))
            else:
                mean_rt = np.nan
                std_rt = np.nan
                total_rt = np.nan

            if not np.isnan(mean_rel_unc) and mean_rel_unc > 0 and not np.isnan(mean_rt) and mean_rt > 0:
                fom = 1.0 / ((mean_rel_unc ** 2) * mean_rt)
            else:
                fom = np.nan

            eval_summary[display_name] = {
                "mean_flux_voxel_avg": global_mean_flux,
                "std_flux_voxel_avg": global_std_flux,
                "max_flux": max_voxel_mean,
                "max_std_flux": max_voxel_std,
                "mean_rel_unc": mean_rel_unc,
                "mean_runtime_s": mean_rt,
                "std_runtime_s": std_rt,
                "total_runtime_s": total_rt,
                "fom": fom,
            }

            log_and_print(f"--- {display_name} ---")
            log_and_print(f"  Flux (domain avg mean +- std): {global_mean_flux:.6e} +- {global_std_flux:.6e}")
            log_and_print(f"  Flux (peak voxel mean / std): {max_voxel_mean:.6e} / {max_voxel_std:.6e}")
            log_and_print(f"  Mean relative uncertainty (R): {mean_rel_unc:.4f}")
            if rt_list:
                log_and_print(f"  Runtime per run (mean +- std): {mean_rt:.2f} s +- {std_rt:.2f} s (total: {total_rt:.2f} s)")
                log_and_print(f"  Figure of Merit (1 / (R^2*T)): {fom:.4e}")
            if mfp_by_method[display_name]:
                mfp_arr = np.array(mfp_by_method[display_name], dtype=float)
                log_and_print(f"  Measured MFP (mean):           {np.mean(mfp_arr):.4f}")
            log_and_print("-" * 50)

        stats_file = os.path.join(output_dir, f"evaluation_stats_runs_{run_range_str}.npz")
        np.savez_compressed(stats_file, **out_stats)

    for _, data in loaded_runs:
        data.close()

    return eval_summary


if __name__ == "__main__":
    num_runs_to_evaluate = 3
    evaluate_runs(range(1, num_runs_to_evaluate + 1), output_dir="simulation_results")