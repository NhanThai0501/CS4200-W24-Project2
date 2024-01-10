import random
import time


def input_board_size():
    # Prompt the user for the number of queens
    return int(input("Enter the number of queens: "))


def print_board(chromosome):
    # Print the board based on a chromosome
    n = len(chromosome)
    print('Board:')
    for row in range(n):
        line = ''.join('Q ' if chromosome[col] == row else '- ' for col in range(n))
        print(line)
    print('---------------------')


def generate_initial_population(pop_size, n):
    # Generate initial population of random solutions
    return [random.sample(range(n), n) for _ in range(pop_size)]


def calculate_fitness(chromosome):
    # Calculate fitness: number of non-attacking pairs
    n = len(chromosome)
    non_attacking_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(chromosome[i] - chromosome[j]) != j - i:
                non_attacking_pairs += 1
    return non_attacking_pairs


def select(population, fitnesses, selection_size):
    # Select chromosomes for the next generation
    sorted_population = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
    return [chromosome for chromosome, _ in sorted_population[:selection_size]]


def crossover(parent1, parent2):
    # Perform crossover between two parents
    n = len(parent1)
    crossover_point = random.randint(0, n - 1)
    return parent1[:crossover_point] + parent2[crossover_point:]


def mutate(chromosome, mutation_rate):
    # Perform mutation on a chromosome
    n = len(chromosome)
    if random.random() < mutation_rate:
        i, j = random.sample(range(n), 2)
        chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
    return chromosome


def genetic_algorithm(n, pop_size=100, max_generations=1000, mutation_rate=0.05):
    # Run the genetic algorithm to solve the N-Queens problem
    population = generate_initial_population(pop_size, n)
    best_fitness = 0

    for generation in range(max_generations):
        fitnesses = [calculate_fitness(chromosome) for chromosome in population]
        best_fitness = max(fitnesses)
        if best_fitness == n * (n - 1) // 2:  # All pairs are non-attacking
            break

        selected = select(population, fitnesses, pop_size // 2)
        next_generation = []

        while len(next_generation) < pop_size:
            parent1, parent2 = random.sample(selected, 2)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            next_generation.append(child)

        population = next_generation

    best_solution = population[fitnesses.index(best_fitness)]
    return best_solution, best_fitness


# Main execution
# board_size = input_board_size()
board_size = 8
total_iterations = 5  # Set the total number of iterations
total_successes = 0
total_runtime = 0

print('CS4200 Project 2: N-Queens with n = 8')
print('Genetic Algorithm:')

for iteration in range(total_iterations):
    start_time = time.time()
    print(f'\nIteration #{iteration + 1}:')

    initial_population = generate_initial_population(1, board_size)
    initial_board = initial_population[0]
    print("Initial Board:")
    print_board(initial_board)

    solution, fitness_score = genetic_algorithm(board_size)
    end_time = time.time()
    runtime = end_time - start_time
    total_runtime += runtime

    print("Final Board (Solution):")
    print_board(solution)
    print("Fitness score:", fitness_score)

    if fitness_score == board_size * (board_size - 1) // 2:  # Check for success
        total_successes += 1

average_runtime = total_runtime / total_iterations
success_rate = total_successes / total_iterations

print("\n-----REPORT STATISTICS----------")
print(f'N-Queens -> n = {board_size}')
print(f'Number of iterations: {total_iterations}')
print(f'Success Case: {total_successes} / {total_iterations}')
print(f'Success rate: {success_rate * 100:.2f}%')
print(f'Average runtime per iteration: {average_runtime:.8f} seconds')
print("-----END REPORT STATISTICS-----")

import random
import time


def input_board_size():
    # Prompt the user for the number of queens
    return int(input("Enter the number of queens: "))


def print_board(chromosome):
    # Print the board based on a chromosome
    n = len(chromosome)
    print('Board:')
    for row in range(n):
        line = ''.join('Q ' if chromosome[col] == row else '- ' for col in range(n))
        print(line)
    print('---------------------')


def generate_initial_population(pop_size, n):
    # Generate initial population of random solutions
    return [random.sample(range(n), n) for _ in range(pop_size)]


def calculate_fitness(chromosome):
    # Calculate fitness: number of non-attacking pairs
    n = len(chromosome)
    non_attacking_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(chromosome[i] - chromosome[j]) != j - i:
                non_attacking_pairs += 1
    return non_attacking_pairs


def select(population, fitnesses, selection_size):
    # Select chromosomes for the next generation
    sorted_population = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
    return [chromosome for chromosome, _ in sorted_population[:selection_size]]


def crossover(parent1, parent2):
    # Perform crossover between two parents
    n = len(parent1)
    crossover_point = random.randint(0, n - 1)
    return parent1[:crossover_point] + parent2[crossover_point:]


def mutate(chromosome, mutation_rate):
    # Perform mutation on a chromosome
    n = len(chromosome)
    if random.random() < mutation_rate:
        i, j = random.sample(range(n), 2)
        chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
    return chromosome


def genetic_algorithm(n, pop_size=750, max_generations=1000, mutation_rate=0.5):
    # Run the genetic algorithm to solve the N-Queens problem
    population = generate_initial_population(pop_size, n)
    best_fitness = 0

    for generation in range(max_generations):
        fitnesses = [calculate_fitness(chromosome) for chromosome in population]
        best_fitness = max(fitnesses)
        if best_fitness == n * (n - 1) // 2:  # All pairs are non-attacking
            break

        selected = select(population, fitnesses, pop_size // 2)
        next_generation = []

        while len(next_generation) < pop_size:
            parent1, parent2 = random.sample(selected, 2)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            next_generation.append(child)

        population = next_generation

    best_solution = population[fitnesses.index(best_fitness)]
    return best_solution, best_fitness


# Main execution
# board_size = input_board_size()
board_size = 8
total_iterations = 5  # Set the total number of iterations
total_successes = 0
total_runtime = 0

print('CS4200 Project 2: N-Queens with n = 8')
print('Genetic Algorithm:')

for iteration in range(total_iterations):
    start_time = time.time()
    print(f'\nIteration #{iteration + 1}:')

    initial_population = generate_initial_population(1, board_size)
    initial_board = initial_population[0]
    print("Initial Board:")
    print_board(initial_board)

    solution, fitness_score = genetic_algorithm(board_size)
    end_time = time.time()
    runtime = end_time - start_time
    total_runtime += runtime

    print("Final Board (Solution):")
    print_board(solution)
    print("Fitness score:", fitness_score)

    if fitness_score == board_size * (board_size - 1) // 2:  # Check for success
        total_successes += 1

average_runtime = total_runtime / total_iterations
success_rate = total_successes / total_iterations

print("\n-----REPORT STATISTICS----------")
print(f'N-Queens -> n = {board_size}')
print(f'Number of iterations: {total_iterations}')
print(f'Success Case: {total_successes} / {total_iterations}')
print(f'Success rate: {success_rate * 100:.2f}%')
print(f'Average runtime per iteration: {average_runtime:.8f} seconds')
print("-----END REPORT STATISTICS-----")

