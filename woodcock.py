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
    def __init__(self, num_voxels, voxel_size):
        self.num_voxels = np.array(num_voxels, dtype=int)
        self.voxel_size = np.array(voxel_size, dtype=float)
        self.total_size = self.num_voxels * self.voxel_size
        self.voxel_volume = np.prod(self.voxel_size)
        # A rács minden voxeljéhez hozzárendeljük a szórási és abszorpciós keresztmetszeteket
        random_max_t = np.random.uniform(0.1,0.6)
        random_max_a = np.random.uniform(0.05,0.2)
        self.sigma_t = np.random.uniform(0.1, random_max_t, size=self.num_voxels)
        self.sigma_a = np.random.uniform(0.05, random_max_a, size=self.num_voxels)
        
        self.sigma_maj = np.max(self.sigma_t)

        self.real_collisions = np.zeros(self.num_voxels, dtype=int)
        
        self.flux_tally = np.zeros(self.num_voxels, dtype=float)

    def which_voxel(self, position):
        voxel_idx = np.floor(position / self.voxel_size).astype(int)
        if np.any(voxel_idx < 0) or np.any(voxel_idx >= self.num_voxels):
            raise ValueError(f"Position {position} is out of bounds for the grid with total size {self.total_size}.")
        return tuple(voxel_idx)

def woodcock_tracking(particle, grid, global_real_paths):
    while particle.state == 'active':
        mean_free_path = 1.0 / grid.sigma_maj
        random_distance = np.random.exponential(scale=mean_free_path)
        particle.path_since_last_real_collision += random_distance
        
        current_voxel = grid.which_voxel(particle.position)
        particle.position += particle.direction * random_distance
        
        if np.any(particle.position < 0) or np.any(particle.position >= grid.total_size):
            particle.state = 'escaped'
            break

        collision_voxel = grid.which_voxel(particle.position)
        grid.flux_tally[collision_voxel] += 1.0 / grid.sigma_maj
        sigma_t_local = grid.sigma_t[collision_voxel]
        probability_of_real = sigma_t_local / grid.sigma_maj
        
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

def get_distance_to_boundary(position, direction, voxel_size):
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

def surface_tracking(particle, grid, global_real_paths):
    optical_path = -np.log(np.random.rand())
    while optical_path > 0 and particle.state == 'active':
        current_voxel = grid.which_voxel(particle.position)
        d_bound = get_distance_to_boundary(particle.position, particle.direction, grid.voxel_size)
        
        if optical_path / grid.sigma_t[current_voxel] < d_bound:
            step_dist = optical_path / grid.sigma_t[current_voxel]
            particle.position += particle.direction * step_dist
            particle.path_since_last_real_collision += step_dist
            grid.flux_tally[current_voxel] += step_dist
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
    while particle.state == 'active':
        dist_to_collide = np.random.exponential(scale=1.0 / grid.sigma_maj)
        while dist_to_collide > 0 and particle.state == 'active':

            current_voxel = grid.which_voxel(particle.position)
            d_bound = get_distance_to_boundary(particle.position, particle.direction, grid.voxel_size)

            if dist_to_collide < d_bound:
                step_dist = dist_to_collide
                particle.path_since_last_real_collision += step_dist
                particle.position += particle.direction * step_dist
                grid.flux_tally[current_voxel] += step_dist

                dist_to_collide = 0.0  # A részecske elérte az ütközést

                sigma_t_local = grid.sigma_t[current_voxel]
                probability_of_real = sigma_t_local / grid.sigma_maj
                
                if np.random.rand() < probability_of_real:
                    global_real_paths.append(particle.path_since_last_real_collision)
                    particle.path_since_last_real_collision = 0.0
                    
                    prob_absorption = grid.sigma_a[current_voxel] / sigma_t_local
                    if np.random.rand() < prob_absorption:
                        particle.state = 'absorbed'
                    else:
                        particle.direction = isotropic_dir()
                else:
                    particle.virtual_collisions += 1
                    
            else:
                # A részecske elmegy a határig, plusz egy nagyon pici ráhagyás, hogy átlépjen az új voxelbe
                step_dist = d_bound + 1e-6 
                particle.path_since_last_real_collision += step_dist
                particle.position += particle.direction * step_dist
                if np.any(particle.position < 0) or np.any(particle.position >= grid.total_size):
                    particle.state = 'escaped'
                    break
                    
                grid.flux_tally[current_voxel] += step_dist
                dist_to_collide -= step_dist

def compare_methods(N_particles=2000):
    methods = [
        ("Felületkövetés (Surface)", surface_tracking),
        ("Woodcock-követés (Delta)", woodcock_tracking),
        ("Kombinált (Hibrid) követés", combined_tracking)
    ]
    print(f"=== SZIMULÁCIÓ INDÍTÁSA ({N_particles} részecske/módszer) ===")
    
    fluxes = {}
    shared_grid = Grid(num_voxels=[50, 50, 50], voxel_size=[1.0, 1.0, 1.0])
    for name, track_func in methods:
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
            
        mean_path = np.mean(global_real_paths)
        scalar_flux = shared_grid.flux_tally / (N_particles * shared_grid.voxel_volume)
        fluxes[name] = scalar_flux
        
        print(f"--- {name} ---")
        print(f"Mért átlagos szabadúthossz: {mean_path:.3f}")
        print(f"Státusz: {escaped_count} kiszökött | {absorbed_count} elnyelődött")
        print(f"Virtuális ütközések száma: {virtual_count}")
        t1 = time.time()
        print(f"Simulation time: {t1 - t0:.2f} seconds")
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

compare_methods(N_particles=500000)