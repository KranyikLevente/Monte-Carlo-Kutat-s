import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import time
import os
import csv


def isotropic_dir():
    """ Generates a normalized isotropic direction vector. """
    cos_theta = 2 * np.random.rand() - 1
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    phi = 2 * np.pi * np.random.rand()
    return np.array([
        sin_theta * np.cos(phi),
        sin_theta * np.sin(phi),
        cos_theta
    ])

class Particle:
    def __init__(self, position, direction, state='active'):
        self.position = np.array(position, dtype=float)
        self.direction = np.array(direction, dtype=float)
        norm = np.linalg.norm(self.direction)
        if norm > 0:
            self.direction /= norm
        else:
            self.direction = isotropic_dir()
        self.state = state
        self.virtual_collisions = 0
        self.path_since_last_real_collision = 0.0

class Grid:
    def __init__(self, num_voxels, voxel_size, macro_block_size=5):
        self.num_voxels = np.array(num_voxels, dtype=int)
        self.voxel_size = np.array(voxel_size, dtype=float)
        self.total_size = self.num_voxels * self.voxel_size
        self.voxel_volume = np.prod(self.voxel_size)

        self.real_collisions = np.zeros(self.num_voxels, dtype=int)
        self.flux_tally = np.zeros(self.num_voxels, dtype=float)

        random_max_t = np.random.uniform(0.1,0.4)
        random_max_a = np.random.uniform(0.05,0.15)
        
        self.sigma_t = np.random.uniform(0.1, random_max_t, size=self.num_voxels) 
        self.sigma_a = np.random.uniform(0.05, random_max_a, size=self.num_voxels)

        self.macro_block_size = int(macro_block_size)
        self.macro_num_blocks = self.num_voxels // self.macro_block_size
        self.macro_flux_tally = np.zeros(self.macro_num_blocks, dtype=float)

        self.sigma_maj = np.zeros(self.macro_num_blocks, dtype=float) 
        for i in range(self.macro_num_blocks[0]):
            for j in range(self.macro_num_blocks[1]):
                for k in range(self.macro_num_blocks[2]):
                            
                            i_start, i_end = i * self.macro_block_size, (i + 1) * self.macro_block_size
                            j_start, j_end = j * self.macro_block_size, (j + 1) * self.macro_block_size
                            k_start, k_end = k * self.macro_block_size, (k + 1) * self.macro_block_size
                            
                            block_sigma_t = self.sigma_t[i_start:i_end, j_start:j_end, k_start:k_end]
                            
                            self.sigma_maj[i, j, k] = np.max(block_sigma_t)
        self.sigma_maj_max = np.max(self.sigma_maj)

    def which_micro_voxel(self, position):
        voxel_idx = np.floor(position / self.voxel_size).astype(int)
        if np.any(voxel_idx < 0) or np.any(voxel_idx >= self.num_voxels):
            raise ValueError(f"Position {position} is out of bounds for the micro grid with total size {self.total_size}.")
            
        return tuple(voxel_idx)

    def which_macro_voxel(self, position):
        macro_size = self.voxel_size * self.macro_block_size
        voxel_idx = np.floor(position / macro_size).astype(int)
        if np.any(voxel_idx < 0) or np.any(voxel_idx >= self.macro_num_blocks):
            raise ValueError(f"Position {position} is out of bounds for the macro grid.")
            
        return tuple(voxel_idx)
    def get_distance_to_boundary(self, position, direction, voxel_size):
        d_min = np.inf
        for i in range(3):
            if direction[i] > 0:
                boundary = (np.floor(position[i] / voxel_size[i]) + 1) * voxel_size[i]
                d = (boundary - position[i]) / direction[i]
                if 0 < d < d_min: 
                    d_min = d
            elif direction[i] < 0:
                boundary = np.floor(position[i] / voxel_size[i]) * voxel_size[i]
                if np.isclose(position[i], boundary):  
                    boundary -= voxel_size[i]
                d = (boundary - position[i]) / direction[i]
                if 0 < d < d_min: 
                    d_min = d
        return d_min
    
def woodcock_tracking(particle, grid, global_real_paths):
    while particle.state == 'active':
        mean_free_path = 1.0 / grid.sigma_maj_max
        random_distance = np.random.exponential(scale=mean_free_path)

        particle.path_since_last_real_collision += random_distance
        particle.position += particle.direction * random_distance
        
        if np.any(particle.position < 0) or np.any(particle.position >= grid.total_size):
            particle.state = 'escaped'
            break

        collision_voxel = grid.which_micro_voxel(particle.position)
        grid.flux_tally[collision_voxel] += random_distance
        
        sigma_t_local = grid.sigma_t[collision_voxel]
        probability_of_real = sigma_t_local / grid.sigma_maj_max
        
        if np.random.rand() < probability_of_real:
            grid.real_collisions[collision_voxel] += 1
            
            # Valós ütközés történt, elmentjük a két ütközés közti utat, majd nullázzuk
            global_real_paths.append(particle.path_since_last_real_collision)
            particle.path_since_last_real_collision = 0.0
            
            sigma_a_local = grid.sigma_a[collision_voxel]
            prob_absorption = sigma_a_local / sigma_t_local
            
            if np.random.rand() < prob_absorption:
                particle.state = 'absorbed'
            else:
                particle.direction = isotropic_dir()
        else:
            particle.virtual_collisions += 1

def surface_tracking(particle, grid, global_real_paths):
    optical_path = -np.log(np.random.rand())
    while optical_path > 0 and particle.state == 'active':
        current_voxel = grid.which_micro_voxel(particle.position)
        d_bound = grid.get_distance_to_boundary(particle.position, particle.direction, grid.voxel_size)
        
        if optical_path / grid.sigma_t[current_voxel] < d_bound:
            step_dist = optical_path / grid.sigma_t[current_voxel]
            particle.position += particle.direction * step_dist
            particle.path_since_last_real_collision += step_dist
            grid.flux_tally[current_voxel] += step_dist
            grid.real_collisions[current_voxel] += 1
            global_real_paths.append(particle.path_since_last_real_collision)
            particle.path_since_last_real_collision = 0.0
            
            if np.random.rand() < grid.sigma_a[current_voxel] / grid.sigma_t[current_voxel]:
                particle.state = 'absorbed'
            else:
                particle.direction = isotropic_dir()
                optical_path = -np.log(np.random.rand())
        else:
            step_dist = d_bound + 1e-6
            particle.position += particle.direction * step_dist
            particle.path_since_last_real_collision += step_dist
            grid.flux_tally[current_voxel] += step_dist
            
            optical_path -= step_dist * grid.sigma_t[current_voxel]
            if np.any(particle.position < 0) or np.any(particle.position >= grid.total_size):
                particle.state = 'escaped'
                break

def combined_woodcock_tracking(particle, grid, global_real_paths):
    macro_size = grid.voxel_size * grid.macro_block_size
    
    while particle.state == 'active':
        current_macro = grid.which_macro_voxel(particle.position)
        sigma_maj_local = grid.sigma_maj[current_macro]
        
        d_woodcock = -np.log(np.random.rand()) / sigma_maj_local
        d_macro_bound = grid.get_distance_to_boundary(particle.position, particle.direction, macro_size)
        
        if d_woodcock < d_macro_bound:
            particle.position += particle.direction * d_woodcock
            particle.path_since_last_real_collision += d_woodcock
            
            current_micro = grid.which_micro_voxel(particle.position)
            
            grid.flux_tally[current_micro] += d_woodcock
            
            sigma_t_local = grid.sigma_t[current_micro]
            probability_of_real = sigma_t_local / sigma_maj_local
            
            if np.random.rand() < probability_of_real:
                grid.real_collisions[current_micro] += 1
                global_real_paths.append(particle.path_since_last_real_collision)
                particle.path_since_last_real_collision = 0.0
                
                prob_absorption = grid.sigma_a[current_micro] / sigma_t_local
                if np.random.rand() < prob_absorption:
                    particle.state = 'absorbed'
                else:
                    particle.direction = isotropic_dir()
            else:
                particle.virtual_collisions += 1
        else:
            step_dist = d_macro_bound + 1e-6
            current_micro = grid.which_micro_voxel(particle.position)
            grid.flux_tally[current_micro] += step_dist
            particle.position += particle.direction * step_dist
            particle.path_since_last_real_collision += step_dist
            
            if np.any(particle.position < 0) or np.any(particle.position >= grid.total_size):
                particle.state = 'escaped'
                break

def combined_surface_tracking(particle, grid, global_real_paths):
    optical_path = -np.log(np.random.rand())
    macro_size = grid.voxel_size * grid.macro_block_size
    
    while optical_path > 0 and particle.state == 'active':
        current_macro = grid.which_macro_voxel(particle.position)
        d_bound = grid.get_distance_to_boundary(particle.position, particle.direction, macro_size)
        
        if optical_path / grid.sigma_maj[current_macro] < d_bound:
            step_dist = optical_path / grid.sigma_maj[current_macro]
            particle.position += particle.direction * step_dist
            
            current_micro = grid.which_micro_voxel(particle.position)
            probability_of_real = grid.sigma_t[current_micro] / grid.sigma_maj[current_macro]
            
            if np.random.rand() < probability_of_real:
                particle.path_since_last_real_collision += step_dist
                grid.macro_flux_tally[current_macro] += step_dist
                grid.flux_tally[current_micro] += step_dist
                grid.real_collisions[current_micro] += 1
                global_real_paths.append(particle.path_since_last_real_collision)
                particle.path_since_last_real_collision = 0.0
                
                if np.random.rand() < grid.sigma_a[current_micro] / grid.sigma_t[current_micro]:
                    particle.state = 'absorbed'
                else:
                    particle.direction = isotropic_dir()
                    optical_path = -np.log(np.random.rand())
            else:
                particle.virtual_collisions += 1
                particle.path_since_last_real_collision += step_dist
                grid.flux_tally[current_micro] += step_dist
                optical_path = -np.log(np.random.rand())
        else:
            step_dist = d_bound + 1e-6
            
            current_micro = grid.which_micro_voxel(particle.position)
            grid.flux_tally[current_micro] += step_dist
            
            particle.position += particle.direction * step_dist
            particle.path_since_last_real_collision += step_dist
            
            optical_path -= step_dist * grid.sigma_maj[current_macro]

            if np.any(particle.position < 0) or np.any(particle.position >= grid.total_size):
                particle.state = 'escaped'
                break
"""
def compare_methods(N_particles=2000):
    #numpy array mentése
    methods = [
        ("Felületkövetés (Surface)", surface_tracking),
        ("Woodcock-követés (Delta)", woodcock_tracking),
        ("Kombinált (Hibrid Woodcock) követés", combined_woodcock_tracking),
        ("Kombinált felületkövetés (Hibrid Surface) követés", combined_surface_tracking),
    ]
    print(f"=== SZIMULÁCIÓ INDÍTÁSA ({N_particles} részecske/módszer) ===")
    
    fluxes = {}
    shared_grid = Grid(num_voxels=[50, 50, 50], voxel_size=[1.0, 1.0, 1.0], macro_block_size=5)
    
    num_spikes = 5
    high_sigma_t = 5.0
    bs = shared_grid.macro_block_size
    #ki lehetne plotolni az eredeti sigma_t eloszlást, hogy lássuk a kiugró értékeket
    #micro rácsra is(talán macrora is (majoráns hatáskeresztmetszet miatt))
    
    for _ in range(num_spikes):
        m_x, m_y, m_z = np.random.randint(0, shared_grid.macro_num_blocks, size=3)
        u_x, u_y, u_z = np.random.randint(0, bs, size=3)
        
        glob_x, glob_y, glob_z = m_x * bs + u_x, m_y * bs + u_y, m_z * bs + u_z
        
        shared_grid.sigma_t[glob_x, glob_y, glob_z] = high_sigma_t
        shared_grid.sigma_a[glob_x, glob_y, glob_z] = high_sigma_t * 0.9 
        shared_grid.sigma_maj[m_x, m_y, m_z] = high_sigma_t

    shared_grid.sigma_maj_max = np.max(shared_grid.sigma_maj)

    for name, track_func in methods:
        shared_grid.flux_tally.fill(0.0)
        shared_grid.real_collisions.fill(0)
        shared_grid.macro_flux_tally.fill(0)
        
        t0 = time.time()
        global_real_paths = []
        escaped_count = 0
        absorbed_count = 0
        virtual_count = 0
        
        for i in range(N_particles):
            p = Particle(position=[25.5, 25.5, 25.5], direction=isotropic_dir())
            track_func(p, shared_grid, global_real_paths)
            
            if p.state == 'escaped':
                escaped_count += 1
            elif p.state == 'absorbed':
                absorbed_count += 1
            virtual_count += p.virtual_collisions
            
        mean_path = np.mean(global_real_paths) if global_real_paths else 0.0
        scalar_flux = shared_grid.flux_tally / (N_particles * shared_grid.voxel_volume)
        fluxes[name] = scalar_flux
        if "Hibrid Surface" in name:
            macro_vol = np.prod(shared_grid.voxel_size * shared_grid.macro_block_size)
            fluxes["Makro Hibrid Surface"] = shared_grid.macro_flux_tally / (N_particles * macro_vol)
        
        print(f"--- {name} ---")
        print(f"Mért átlagos szabadúthossz: {mean_path:.3f}")
        print(f"Státusz: {escaped_count} kiszökött | {absorbed_count} elnyelődött")
        print(f"Virtuális ütközések száma: {virtual_count}")
        t1 = time.time()
        print(f"Futási idő: {t1 - t0:.2f} másodperc")
        print("-" * 50)
        
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle('Fluxustérképek összehasonlítása (logaritmikus skála)', fontsize=16)
    #numpy array mentése
    for ax, (name, flux) in zip(axes, fluxes.items()):
        slice_2d = flux[:, :, flux.shape[2] // 2]
        
        max_flux = np.max(slice_2d)
        if max_flux <= 1e-3:
            max_flux = 1e-2
        # fluxus normalizálás részcske számmal oszlás és voxel térfogattal
        #slice_2d /= (N_particles * shared_grid.voxel_volume)    
        im = ax.imshow(slice_2d, origin='lower', cmap='viridis', norm=LogNorm(vmin=1e-3, vmax=max_flux))
        
        ax.set_title(name, fontsize=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Fluxus')
        
    plt.tight_layout()
    plt.show()
"""


def create_shared_grid():
    shared_grid = Grid(num_voxels=[50, 50, 50], voxel_size=[1.0, 1.0, 1.0], macro_block_size=5)
    
    num_spikes = 0
    high_sigma_t = 5.0
    bs = shared_grid.macro_block_size
    
    for _ in range(num_spikes):
        m_x, m_y, m_z = np.random.randint(0, shared_grid.macro_num_blocks, size=3)
        u_x, u_y, u_z = np.random.randint(0, bs, size=3)
        
        glob_x, glob_y, glob_z = m_x * bs + u_x, m_y * bs + u_y, m_z * bs + u_z
        
        shared_grid.sigma_t[glob_x, glob_y, glob_z] = high_sigma_t
        shared_grid.sigma_a[glob_x, glob_y, glob_z] = high_sigma_t * 0.9 
        shared_grid.sigma_maj[m_x, m_y, m_z] = high_sigma_t

    shared_grid.sigma_maj_max = np.max(shared_grid.sigma_maj)
    return shared_grid
    
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os

def apply_rod(grid, m_x, m_y, high_sigma_t=5.0, rod_size=3):
    """
    Places a 'rod' with a 3x3 cross-section extending along the entire Z-axis.
    No Z coordinate is needed as it spans the entire model.
    """
    bs = grid.macro_block_size
    base_x, base_y = m_x * bs, m_y * bs
    
    offset = (bs - rod_size) // 2 
    x0, x1 = base_x + offset, base_x + offset + rod_size
    y0, y1 = base_y + offset, base_y + offset + rod_size
    
    # The ':' means the rod spans the entire Z-axis of the model
    grid.sigma_t[x0:x1, y0:y1, :] = high_sigma_t
    grid.sigma_a[x0:x1, y0:y1, :] = high_sigma_t * 0.9 

def recalculate_macro_grid(grid):
    """ Helper function: Recalculate macro maximums after placing the rods. """
    bs = grid.macro_block_size
    for i in range(grid.macro_num_blocks[0]):
        for j in range(grid.macro_num_blocks[1]):
            for k in range(grid.macro_num_blocks[2]):
                block = grid.sigma_t[i*bs:(i+1)*bs, j*bs:(j+1)*bs, k*bs:(k+1)*bs]
                grid.sigma_maj[i, j, k] = np.max(block)
    grid.sigma_maj_max = np.max(grid.sigma_maj)


def create_grid_edge_rods():
    """ CASE 1: 10 rods placed at the edges of the model. """
    grid = Grid(num_voxels=[50, 50, 50], voxel_size=[1.0, 1.0, 1.0], macro_block_size=5)
    
    edge_macros_xy = [
        (0, 0), (9, 9), (0, 9), (9, 0),  # Corners
        (5, 0), (0, 5), (5, 9), (9, 5),  # Middle of the edges
        (2, 0), (7, 0)                   # Extra points
    ]
    
    for m_x, m_y in edge_macros_xy:
        apply_rod(grid, m_x, m_y)
        
    recalculate_macro_grid(grid)
    return grid


def create_grid_inner_rods():
    """ CASE 2: 6-7 rods placed near the center of the model (within a 15-voxel radius). """
    grid = Grid(num_voxels=[50, 50, 50], voxel_size=[1.0, 1.0, 1.0], macro_block_size=5)
    
    center = 4.5
    radius_in_macros = 3.0 
    
    inner_candidates = []
    for i in range(10):
        for j in range(10):
            distance = np.sqrt((i - center)**2 + (j - center)**2)
            if distance <= radius_in_macros:
                inner_candidates.append((i, j))
    
    np.random.seed(42)
    np.random.shuffle(inner_candidates)
    
    for m_x, m_y in inner_candidates[:7]:
        apply_rod(grid, m_x, m_y)
        
    recalculate_macro_grid(grid)
    return grid


def create_grid_systematic_rods():
    """ CASE 3: Checkerboard arrangement (only in the X-Y plane). """
    grid = Grid(num_voxels=[50, 50, 50], voxel_size=[1.0, 1.0, 1.0], macro_block_size=5)
    
    np.random.seed(42)
    for i in range(10):
        for j in range(10):
            if (i + j) % 2 == 0:
                if np.random.rand() < 0.66:
                    apply_rod(grid, i, j)
                        
    recalculate_macro_grid(grid)
    return grid

def create_grid_random_rods(num_rods=5):
    """ 
    CASE 4: Randomly placed rods across the X-Y plane. 
    You can easily change the number of rods by passing a different num_rods value.
    """
    grid = Grid(num_voxels=[20,20,20], voxel_size=[2.5, 2.5, 2.5], macro_block_size=2)
    
    # Optional: Set a seed if you want the "random" rods to be in the exact 
    # same place every time you run the code for testing. 
    # np.random.seed(42) 
    
    for _ in range(num_rods):
        # Choose a random macroblock in the X and Y directions
        m_x = np.random.randint(0, grid.macro_num_blocks[0])
        m_y = np.random.randint(0, grid.macro_num_blocks[1])
        
        # Place the rod in the chosen macroblock
        apply_rod(grid, m_x, m_y, high_sigma_t=5.0, rod_size=2)
        
    # Recalculate the macro grid maximums after placing all rods
    recalculate_macro_grid(grid)
    return grid

def save_and_plot_initial_grid(grid, output_dir):
    """ Plotting with LogNorm so the random background noise is also visible. """
    np.savez_compressed(
        os.path.join(output_dir, "original_grid.npz"),
        sigma_t=grid.sigma_t,
        sigma_a=grid.sigma_a,
        sigma_maj=grid.sigma_maj,
        num_voxels=grid.num_voxels,
        voxel_size=grid.voxel_size,
        macro_block_size=np.array(grid.macro_block_size)
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Initial Cross-Section Distribution', fontsize=15, fontweight='bold')
    
    # Any slice is fine because the rods are constant in the Z-direction
    mid_z_micro = grid.sigma_t.shape[2] // 2
    mid_z_macro = grid.sigma_maj.shape[2] // 2
    
    # Using LogNorm for the color scale
    norm_micro = LogNorm(vmin=np.min(grid.sigma_t), vmax=np.max(grid.sigma_t))
    norm_macro = LogNorm(vmin=np.min(grid.sigma_maj), vmax=np.max(grid.sigma_maj))
    
    im0 = axes[0].imshow(grid.sigma_t[:, :, mid_z_micro], origin='lower', cmap='inferno', norm=norm_micro)
    axes[0].set_title(r'Micro Grid ($\Sigma_t$)', fontsize=13)
    axes[0].set_xlabel('X [voxel]')
    axes[0].set_ylabel('Y [voxel]')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    im1 = axes[1].imshow(grid.sigma_maj[:, :, mid_z_macro], origin='lower', cmap='inferno', norm=norm_macro)
    axes[1].set_title(r'Macro Grid ($\Sigma_{maj}$)', fontsize=13)
    axes[1].set_xlabel('X [macroblock]')
    axes[1].set_ylabel('Y [macroblock]')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "original_grid_sigma.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

def run_experiments(N_particles=5000, num_runs=10, output_dir="simulation_results"):
    os.makedirs(output_dir, exist_ok=True)
    
    shared_grid = create_grid_random_rods(5)
    save_and_plot_initial_grid(shared_grid, output_dir)
    
    methods = [
        ("Surface", "Surface Tracking", surface_tracking),
        ("Woodcock", "Woodcock Tracking (Delta)", woodcock_tracking),
        ("Hybrid_Woodcock", "Hybrid Woodcock Tracking", combined_woodcock_tracking),
        ("Hybrid_Surface", "Hybrid Surface Tracking", combined_surface_tracking),
    ]
    
    log_file_path = os.path.join(output_dir, "simulation_logs.txt")
    csv_file_path = os.path.join(output_dir, "summary_metrics.csv")
    
    with open(log_file_path, "w", encoding="utf-8") as log_file, \
         open(csv_file_path, "w", newline="", encoding="utf-8") as csv_file:
        
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Run", "Method_ID", "Method_Name", "Mean_Free_Path", "Escaped", "Absorbed", "Virtual_Collisions", "Runtime_s"])
        
        def log_and_print(text):
            print(text)
            log_file.write(text + "\n")

        for run_idx in range(1, num_runs + 1):
            log_and_print(f"\n================ RUN {run_idx}/{num_runs} ({N_particles} particles/method) ================")
            
            fluxes_for_plot = {}
            arrays_to_save = {}
            
            for short_name, display_name, track_func in methods:
                shared_grid.flux_tally.fill(0.0)
                shared_grid.real_collisions.fill(0)
                shared_grid.macro_flux_tally.fill(0.0)
                
                t0 = time.time()
                global_real_paths = []
                escaped_count = 0
                absorbed_count = 0
                virtual_count = 0
                
                for _ in range(N_particles):
                    start_pos = np.random.uniform(20.5, 30.5, size=3)
                    p = Particle(position=start_pos, direction=isotropic_dir())
                    track_func(p, shared_grid, global_real_paths)
                    
                    if p.state == 'escaped':
                        escaped_count += 1
                    elif p.state == 'absorbed':
                        absorbed_count += 1
                    virtual_count += p.virtual_collisions
                    
                t1 = time.time()
                runtime = t1 - t0
                
                mean_path = np.mean(global_real_paths) if global_real_paths else 0.0
                scalar_flux = (shared_grid.flux_tally / (N_particles * shared_grid.voxel_volume)).copy()
                
                fluxes_for_plot[display_name] = scalar_flux
                arrays_to_save[f"{short_name}_flux"] = scalar_flux
                arrays_to_save[f"{short_name}_real_collisions"] = shared_grid.real_collisions.copy()
                arrays_to_save[f"{short_name}_real_paths"] = np.array(global_real_paths)
                
                if short_name == "Hybrid_Surface":
                    macro_vol = np.prod(shared_grid.voxel_size * shared_grid.macro_block_size)
                    macro_flux = (shared_grid.macro_flux_tally / (N_particles * macro_vol)).copy()
                    fluxes_for_plot["Macro Hybrid Surface"] = macro_flux
                    arrays_to_save["Macro_Hybrid_Surface_flux"] = macro_flux
                
                log_and_print(f"--- {display_name} (Run {run_idx}) ---")
                log_and_print(f"Measured mean free path: {mean_path:.3f}")
                log_and_print(f"Status: {escaped_count} escaped | {absorbed_count} absorbed")
                log_and_print(f"Number of virtual collisions: {virtual_count}")
                log_and_print(f"Runtime: {runtime:.2f} seconds")
                log_and_print("-" * 50)
                
                csv_writer.writerow([run_idx, short_name, display_name, mean_path, escaped_count, absorbed_count, virtual_count, runtime])
            
            npz_filename = os.path.join(output_dir, f"results_run_{run_idx:02d}.npz")
            np.savez_compressed(npz_filename, **arrays_to_save)
            
            fig, axes = plt.subplots(1, 5, figsize=(22, 5))
            fig.suptitle(f'Flux Map Comparison - Run {run_idx} (Logarithmic Scale)', fontsize=16)
            
            for ax, (name, flux) in zip(axes, fluxes_for_plot.items()):
                slice_2d = flux[:, :, flux.shape[2] // 2]
                
                max_flux = np.max(slice_2d)
                if max_flux <= 1e-3:
                    max_flux = 1e-2
                    
                im = ax.imshow(slice_2d, origin='lower', cmap='viridis', norm=LogNorm(vmin=1e-3, vmax=max_flux))
                ax.set_title(name, fontsize=10)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Flux')
                
            plt.tight_layout()
            plot_filename = os.path.join(output_dir, f"flux_comparison_run_{run_idx:02d}.png")
            plt.savefig(plot_filename, dpi=300)
            plt.close(fig)

if __name__ == "__main__":
    run_experiments(N_particles=500000, num_runs=1)