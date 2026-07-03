import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

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
        self.sigma_t = np.full(self.num_voxels, 0.5) 
        self.sigma_a = np.full(self.num_voxels, 0.1)
        
        self.sigma_maj = np.max(self.sigma_t)

        self.real_collisions = np.zeros(self.num_voxels, dtype=int)
        
        self.flux_tally = np.zeros(self.num_voxels, dtype=float)

    def which_voxel(self, position):
        voxel_idx = np.floor(position / self.voxel_size).astype(int)
        voxel_idx = np.clip(voxel_idx, 0, self.num_voxels - 1)
        return tuple(voxel_idx)

def surface_tracking(particle, grid, global_real_paths):
    dist_to_collide = np.random.exponential(scale=1.0 / grid.sigma_t[0, 0, 0])
    while dist_to_collide > 0 and particle.state == 'active':
        current_voxel = grid.which_voxel(particle.position)
        d_bound = get_distance_to_boundary(particle.position, particle.direction, grid.voxel_size)
        
        if dist_to_collide < d_bound:
            step_dist = dist_to_collide
            particle.position += particle.direction * step_dist
            particle.path_since_last_real_collision += step_dist
            grid.flux_tally[current_voxel] += step_dist
            global_real_paths.append(particle.path_since_last_real_collision)
            particle.path_since_last_real_collision = 0.0
            
            if np.random.rand() < grid.sigma_a[current_voxel] / grid.sigma_t[current_voxel]:
                particle.state = 'absorbed'
            else:
                particle.direction = isotropic_dir()
                dist_to_collide = np.random.exponential(scale=1.0 / grid.sigma_t[current_voxel])
        else:
            step_dist = d_bound + 1e-6
            particle.position += particle.direction * step_dist
            particle.path_since_last_real_collision += step_dist
            grid.flux_tally[current_voxel] += step_dist
            
            dist_to_collide -= step_dist 
            if np.any(particle.position < 0) or np.any(particle.position >= grid.total_size):
                particle.state = 'escaped'
                break

def woodcock_tracking(particle, grid, global_real_paths):
    distance_traveled = 0.0
    
    while particle.state == 'active':
        mean_free_path = 1.0 / grid.sigma_maj
        random_distance = np.random.exponential(scale=mean_free_path)
        distance_traveled += random_distance
        particle.path_since_last_real_collision += random_distance
        
        current_voxel = grid.which_voxel(particle.position)
        particle.position += particle.direction * random_distance
        
        if np.any(particle.position < 0) or np.any(particle.position >= grid.total_size):
            particle.state = 'escaped'
            break
        grid.flux_tally[current_voxel] += random_distance
        target_voxel = grid.which_voxel(particle.position)
        sigma_t_local = grid.sigma_t[target_voxel]
        probability_of_real = sigma_t_local / grid.sigma_maj
        
        if np.random.rand() < probability_of_real:
            grid.real_collisions[target_voxel] += 1
            
            # Valós ütközés történt, elmentjük a két ütközés közti utat, majd nullázzuk
            global_real_paths.append(particle.path_since_last_real_collision)
            particle.path_since_last_real_collision = 0.0
            
            sigma_a_local = grid.sigma_a[target_voxel]
            prob_absorption = sigma_a_local / sigma_t_local
            
            if np.random.rand() < prob_absorption:
                particle.state = 'absorbed'
            else:
                particle.direction = isotropic_dir()
        else:
            particle.virtual_collisions += 1

    return distance_traveled

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

def combined_tracking(particle, grid, global_real_paths):
    distance_traveled = 0.0
    
    while particle.state == 'active':
        current_voxel = grid.which_voxel(particle.position)
        
        d_bound = get_distance_to_boundary(particle.position, particle.direction, grid.voxel_size)
        d_woodcock = np.random.exponential(scale=1.0 / grid.sigma_maj)
        # woodcock lépés és a határig tartó lépés összehasonlítása
        if d_woodcock < d_bound:
            step_dist = d_woodcock
            particle.path_since_last_real_collision += step_dist
            particle.position += particle.direction * step_dist
            distance_traveled += step_dist
            
            grid.flux_tally[current_voxel] += step_dist
            
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
            distance_traveled += step_dist
            
            if np.any(particle.position < 0) or np.any(particle.position >= grid.total_size):
                particle.state = 'escaped'
                break
                
            grid.flux_tally[current_voxel] += step_dist

grid = Grid(num_voxels=[10, 10, 10], voxel_size=[1.0, 1.0, 1.0])
N_particles = 1000

escaped_count = 0
absorbed_count = 0
global_real_paths = [] 

for i in range(N_particles):
    direction = isotropic_dir()
    particle = Particle(position=[5.5, 5.5, 5.5], direction=direction)
    woodcock_tracking(particle, grid, global_real_paths)
    
    if particle.state == 'escaped':
        escaped_count += 1
    elif particle.state == 'absorbed':
        absorbed_count += 1

scalar_flux_grid = grid.flux_tally / (N_particles * grid.voxel_volume)

print(f"--- Eredmények {N_particles} részecske után ---")
print(f"Kiszökött: {escaped_count} db | Elnyelődött: {absorbed_count} db")
print(f"Valós ütközések közti átlagos távolság: {np.mean(global_real_paths):.3f}")
print(f"A maximális fluxus értéke a rácsban: {np.max(scalar_flux_grid):.5f}")
print(f"Az átlagos fluxus a rácsban: {np.mean(scalar_flux_grid):.5f}")

# Fluxus vizualizálása logaritmikus skálán
plt.figure(figsize=(8, 6))
plt.imshow(scalar_flux_grid[:, :, scalar_flux_grid.shape[2] // 2], origin='lower', cmap='viridis', norm=LogNorm())
plt.colorbar(label='Fluxus')
plt.title('Rács középső szeletének fluxusa')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

def compare_methods(N_particles=200000):
    methods = [
        ("Felületkövetés (Surface)", surface_tracking),
        ("Woodcock-követés (Delta)", woodcock_tracking),
        ("Kombinált (Hibrid) követés", combined_tracking)
    ]
    test_grid = Grid(num_voxels=[10, 10, 10], voxel_size=[1.0, 1.0, 1.0])
    sigma_t_homogen = test_grid.sigma_t[0, 0, 0] if test_grid.sigma_t.ndim > 1 else test_grid.sigma_t[0]
    expected_mfp = 1.0 / sigma_t_homogen
    print(f"=== SZIMULÁCIÓ INDÍTÁSA ({N_particles} részecske/módszer) ===")
    print(f"Elméleti átlagos szabadúthossz: {expected_mfp:.3f}\n")
    
    fluxes = {}
    
    for name, track_func in methods:
        grid = Grid(num_voxels=[100, 100, 100], voxel_size=[1.0, 1.0, 1.0])
        global_real_paths = []
        escaped_count = 0
        absorbed_count = 0
        virtual_count = 0
        
        for i in range(N_particles):
            p = Particle(position=[50.5, 50.5, 50.5], direction=isotropic_dir())
            track_func(p, grid, global_real_paths)
            
            if p.state == 'escaped':
                escaped_count += 1
            elif p.state == 'absorbed':
                absorbed_count += 1
            virtual_count += p.virtual_collisions
            
        mean_path = np.mean(global_real_paths)
        scalar_flux = grid.flux_tally / (N_particles * grid.voxel_volume)
        fluxes[name] = scalar_flux
        
        print(f"--- {name} ---")
        print(f"Mért átlagos szabadúthossz: {mean_path:.3f} (Hiba: {abs(mean_path-expected_mfp)/expected_mfp*100:.1f}%)")
        print(f"Státusz: {escaped_count} kiszökött | {absorbed_count} elnyelődött")
        print(f"Virtuális ütközések száma: {virtual_count}")
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

compare_methods(N_particles=200000)