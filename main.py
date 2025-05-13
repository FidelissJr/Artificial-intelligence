import numpy as np


def pso(dim = 3, num_particles = 30, max_iter=100, w=0.5, c1=2.05, c2=2.05):
    #inicializa as particulas e as velocidades
    particles = np.random.uniform(0, 1, (num_particles, dim))
    velocities = np.zeros((num_particles, dim))
    
    best_positions = np.copy(particles)

pso()


