import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import time

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

        random_max_t = np.random.uniform(0.1,0.6)
        random_max_a = np.random.uniform(0.05,0.2)

        self.sigma_t = np.random.uniform(0.1, random_max_t, size=self.num_voxels) 
        self.sigma_a = np.random.uniform(0.05, random_max_a, size=self.num_voxels)

        self.macro_block_size = int(macro_block_size)
        self.macro_num_blocks = self.num_voxels // self.macro_block_size

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
        grid.flux_tally[collision_voxel] += 1.0 / grid.sigma_maj_max
        
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

def combined_tracking(particle, grid, global_real_paths):
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
            
            grid.flux_tally[current_micro] += 1.0 / sigma_maj_local
            
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
            particle.position += particle.direction * step_dist
            particle.path_since_last_real_collision += step_dist
            
            if np.any(particle.position < 0) or np.any(particle.position >= grid.total_size):
                particle.state = 'escaped'
                break

def compare_methods(N_particles=2000):
    methods = [
        ("Felületkövetés (Surface)", surface_tracking),
        ("Woodcock-követés (Delta)", woodcock_tracking),
        ("Kombinált (Hibrid) követés", combined_tracking)
    ]
    print(f"=== SZIMULÁCIÓ INDÍTÁSA ({N_particles} részecske/módszer) ===")
    
    fluxes = {}
    shared_grid = Grid(num_voxels=[50, 50, 50], voxel_size=[1.0, 1.0, 1.0], macro_block_size=5)
    
    num_spikes = 5
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

    for name, track_func in methods:
        shared_grid.flux_tally.fill(0.0)
        shared_grid.real_collisions.fill(0)
        
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
        
        print(f"--- {name} ---")
        print(f"Mért átlagos szabadúthossz: {mean_path:.3f}")
        print(f"Státusz: {escaped_count} kiszökött | {absorbed_count} elnyelődött")
        print(f"Virtuális ütközések száma: {virtual_count}")
        t1 = time.time()
        print(f"Futási idő: {t1 - t0:.2f} másodperc")
        print("-" * 50)
        
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Fluxustérképek összehasonlítása (logaritmikus skála)', fontsize=16)
    
    for ax, (name, flux) in zip(axes, fluxes.items()):
        slice_2d = flux[:, :, flux.shape[2] // 2]
        im = ax.imshow(slice_2d, origin='lower', cmap='viridis', norm=LogNorm(vmin=1e-3, vmax=np.max(slice_2d)))
        ax.set_title(name)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Fluxus')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_methods(N_particles=500000)