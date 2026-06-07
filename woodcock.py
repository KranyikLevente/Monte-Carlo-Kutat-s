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
        # Randomly assign total cross-sections (sigma_t) to each voxel
        self.voxel_volume = np.prod(self.voxel_size)
        # Randomly assign total cross-sections and absorption fractions for each voxel
        self.sigma_t = np.random.uniform(0.1, 1.0, size=self.num_voxels)
        absorption_fraction = np.random.uniform(0.1, 0.9, size=self.num_voxels)
        self.sigma_a = self.sigma_t * absorption_fraction
        
        self.sigma_maj = np.max(self.sigma_t)

        self.real_collisions = np.zeros(self.num_voxels, dtype=int)
        
        self.flux_tally = np.zeros(self.num_voxels, dtype=float)

    def which_voxel(self, position):
        voxel_idx = np.floor(position / self.voxel_size).astype(int)
        voxel_idx = np.clip(voxel_idx, 0, self.num_voxels - 1)
        return tuple(voxel_idx)

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