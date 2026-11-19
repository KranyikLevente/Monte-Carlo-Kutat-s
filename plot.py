import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def plot_saved_fluxes(run_index=1, output_dir="simulation_results"):
    npz_filename = os.path.join(output_dir, f"results_run_{run_index:02d}.npz")
    
    if not os.path.exists(npz_filename):
        print(f"Error: Data file not found at {npz_filename}")
        return
        
    data = np.load(npz_filename)
    
    methods = [
        ("Surface_flux", "Surface Tracking"),
        ("Woodcock_flux", "Woodcock Tracking (Delta)"),
        ("Hybrid_Woodcock_flux", "Hybrid Woodcock Tracking"),
        ("Hybrid_Surface_flux", "Hybrid Surface Tracking")
    ]
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(f'Flux Map Comparison - Run {run_index} (Logarithmic Scale)', fontsize=15, fontweight='bold')
    
    for ax, (key, display_name) in zip(axes, methods):
        if key not in data:
            print(f"Warning: {key} not found in data file.")
            continue
            
        flux = data[key]
        
        z_slice = flux.shape[2] // 2
        slice_2d = flux[:, :, z_slice]
        
        max_flux = np.max(slice_2d)
        
        if max_flux <= 0.0:
            max_flux = 1e-9
            vmin = 1e-10
        else:
            vmin = max_flux * 1e-3
            
        im = ax.imshow(slice_2d, origin='lower', cmap='viridis')
        
        ax.set_title(display_name, fontsize=11)
        ax.set_xlabel('X [voxel]')
        ax.set_ylabel('Y [voxel]')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Flux')
        
    plt.tight_layout()
    
    plot_filename = os.path.join(output_dir, f"replot_flux_comparison_run_{run_index:02d}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to: {plot_filename}")
    plt.show()

if __name__ == "__main__":
    plot_saved_fluxes(run_index=1)