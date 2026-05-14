import numpy as np

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

class Grid:
    def __init__(self, num_voxels, voxel_size):
        self.num_voxels = np.array(num_voxels, dtype=int)
        self.voxel_size = np.array(voxel_size, dtype=float)
        self.total_size = self.num_voxels * self.voxel_size
        # Randomly assign total cross-sections (sigma_t) to each voxel
        self.sigma_t = np.random.uniform(0.1, 1.0, size=self.num_voxels)
        # PE: cross sections for absorption should be also specified
        self.sigma_maj = np.max(self.sigma_t)
        
        self.real_collisions = np.zeros(self.num_voxels, dtype=int)
        self.virtual_collisions = np.zeros(self.num_voxels, dtype=int)

    def which_voxel(self, position):
        voxel_idx = np.floor(position / self.voxel_size).astype(int)
        return tuple(voxel_idx)

def woodcock_tracking(particle, grid):
    distance_traveled = 0.0
    
    while particle.state == 'active':
        mean_free_path = 1.0 / grid.sigma_maj
        random_distance = np.random.exponential(scale=mean_free_path)
        distance_traveled += random_distance
        
        particle.position += particle.direction * random_distance
        
        if np.any(particle.position < 0) or np.any(particle.position >= grid.total_size):
            particle.state = 'escaped'
            break
        
        voxel = grid.which_voxel(particle.position)
        
        sigma_t_local = grid.sigma_t[voxel]
        probability_of_real = sigma_t_local / grid.sigma_maj
        
        if np.random.rand() < probability_of_real:
            grid.real_collisions[voxel] += 1
            # PE: collisions can be scatterings as well
            particle.state = 'absorbed'
        else:
            # PE: might be better to count virtual collisions per particle, not per voxel
            grid.virtual_collisions[voxel] += 1

    return distance_traveled

# PE: the voxel/grid size should be appropriate for the cross sections, 
#     i.e. not much  bigger than the mean free path? I did not double check this though 
grid = Grid(num_voxels=[10, 10, 10], voxel_size=[1.0, 1.0, 1.0])

N_particles = 1000
escaped_count = 0
absorbed_count = 0

for i in range(N_particles):
    direction = isotropic_dir()
    particle = Particle(position=[5.5, 5.5, 5.5], direction=direction)
    
    woodcock_tracking(particle, grid)
    
    if particle.state == 'escaped':
        escaped_count += 1
    elif particle.state == 'absorbed':
        absorbed_count += 1

print(f"--- Eredmények {N_particles} részecske után ---")
print(f"Kiszökött a rácsból: {escaped_count} db")
print(f"Elnyelődött (valós ütközést szenvedett): {absorbed_count} db")
print(f"Teljes valós ütközések száma a rácsban: {np.sum(grid.real_collisions)}")
print(f"Teljes virtuális ütközések száma a rácsban: {np.sum(grid.virtual_collisions)}")
