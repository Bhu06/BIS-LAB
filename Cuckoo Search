import numpy as np
import random
import math  # Import the standard math module for gamma function

# Define the objective function for optimization (for data mining, this can represent an error function)
def objective_function(x):
    return np.sum(x ** 2)  # Example: Sphere function (sum of squares) - minimize for best result

# Lévy flight step
def levy_flight(Lambda):
    sigma = (math.gamma(1 + Lambda) * math.sin(math.pi * Lambda / 2) /
             (math.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)
    u = np.random.normal(0, sigma, size=len(bounds))
    v = np.random.normal(0, 1, size=len(bounds))
    step = u / (np.abs(v) ** (1 / Lambda))
    return step

# Cuckoo Search (CS) Algorithm
class CuckooSearch:
    def __init__(self, obj_func, bounds, num_nests=15, pa=0.25, max_iter=100):
        self.obj_func = obj_func
        self.bounds = bounds
        self.num_nests = num_nests  # Number of nests (solutions)
        self.pa = pa                # Discovery probability (probability of abandoning nests)
        self.max_iter = max_iter    # Maximum number of iterations
        
        # Initialize nests (solutions) randomly within bounds
        self.nests = np.random.rand(self.num_nests, len(bounds))
        for i in range(len(bounds)):
            self.nests[:, i] = self.nests[:, i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]
        
        # Evaluate initial fitness
        self.fitness = np.array([self.obj_func(nest) for nest in self.nests])
        self.best_nest = self.nests[np.argmin(self.fitness)]
        self.best_fitness = np.min(self.fitness)

    def run(self):
        for iteration in range(self.max_iter):
            # Generate new solutions via Lévy flights
            new_nests = np.array([self.generate_new_nest(nest) for nest in self.nests])
            
            # Evaluate fitness of new solutions
            new_fitness = np.array([self.obj_func(nest) for nest in new_nests])
            
            # Replace old nests with better solutions
            improvement = new_fitness < self.fitness
            self.fitness[improvement] = new_fitness[improvement]
            self.nests[improvement] = new_nests[improvement]
            
            # Update the best nest if a new one is found
            min_fitness_idx = np.argmin(self.fitness)
            if self.fitness[min_fitness_idx] < self.best_fitness:
                self.best_nest = self.nests[min_fitness_idx]
                self.best_fitness = self.fitness[min_fitness_idx]
            
            # Abandon a fraction of the worst nests and replace them
            self.abandon_worst_nests()
        
        return self.best_nest, self.best_fitness

    def generate_new_nest(self, nest):
        step_size = levy_flight(1.5)  # Using Lévy flight
        new_nest = nest + step_size * (nest - self.best_nest)
        
        # Ensure the new position is within bounds
        for i in range(len(bounds)):
            new_nest[i] = np.clip(new_nest[i], bounds[i][0], bounds[i][1])
        
        return new_nest

    def abandon_worst_nests(self):
        num_abandon = int(self.pa * self.num_nests)
        worst_indices = np.argsort(-self.fitness)[:num_abandon]
        
        for idx in worst_indices:
            # Replace abandoned nests with new random positions
            self.nests[idx] = np.random.rand(len(bounds))
            for i in range(len(bounds)):
                self.nests[idx][i] = self.nests[idx][i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]
            # Update fitness of the newly replaced nests
            self.fitness[idx] = self.obj_func(self.nests[idx])

# Define bounds for the search space (for example, a 2D feature space [-10, 10] for each feature)
bounds = [(-10, 10), (-10, 10)]

# Initialize Cuckoo Search
num_nests = 15
pa = 0.25
max_iter = 100

# Run the Cuckoo Search algorithm
cs = CuckooSearch(objective_function, bounds, num_nests, pa, max_iter)
best_solution, best_fitness = cs.run()
print("Best solution:", best_solution)
print("Best fitness:", best_fitness)
