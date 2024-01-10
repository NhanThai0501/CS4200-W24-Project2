import random
import time


def input_size_board():
    # Take input from user for size of board
    n = input("Enter an integer for the number of queens: ")
    return int(n)


def print_board(board, n):
    # Helper function for printing board
    print('Board:')
    for i in range(len(board)):
        # Print 'Q' for queens (1) and '-' for empty spaces (0) using f-string
        cell_char = 'Q' if board[i] == 1 else '-'
        print(f"{cell_char} ", end='')
        if (i + 1) % n == 0:
            print()
    print('---------------------')
    print(f'Heuristic Value: {find_h_cost(board, n)}')
    print('---------------------')


def generate_random_board(n):
    # Generates a random board for initialization, queens have been calculated row-wise
    generated_board = []
    for i in range(n):
        j = random.randint(0, n - 1)
        row = [0] * n
        row[j] = 1
        generated_board.extend(row)
    return generated_board


def find_collisions(chess_board, board_size):
    # Helper function for calculating queen position collisions
    total_collisions = 0
    collision_details = []
    total_cells = len(chess_board)
    for index in range(total_cells):
        # For each queen on the board, count collisions with other queens, and which kind of collisions they are
        if chess_board[index] == 1:
            for distance in range(1, board_size):
                # Check above current index (north and its diagonals)
                if (index - board_size * distance >= 0):
                    north_index = index - board_size * distance
                    # Direction North
                    if (chess_board[north_index] == 1):
                        total_collisions += 1
                        collision_details.append(f'north: {index} and {north_index}')
                    # Direction Northwest
                    if (int((north_index - distance) / board_size) == int(north_index / board_size)) and (
                            north_index - distance) >= 0:
                        northwest_index = north_index - distance
                        if (chess_board[northwest_index] == 1):
                            total_collisions += 1
                            collision_details.append(f'northwest: {index} and {northwest_index}')
                    # Direction Northeast
                    if (int((north_index + distance) / board_size) == int(north_index / board_size)):
                        northeast_index = north_index + distance
                        if (chess_board[northeast_index] == 1):
                            total_collisions += 1
                            collision_details.append(f'northeast: {index} and {northeast_index}')
                # Check below current index (south and its diagonals)
                if (index + board_size * distance < total_cells):
                    south_index = index + board_size * distance
                    # Direction South
                    if (chess_board[south_index] == 1):
                        total_collisions += 1
                        collision_details.append(f'south: {index} and {south_index}')
                    # Direction Southwest
                    if (int((south_index - distance) / board_size) == int(south_index / board_size)):
                        southwest_index = south_index - distance
                        if (chess_board[southwest_index] == 1):
                            total_collisions += 1
                            collision_details.append(f'southwest: {index} and {southwest_index}')
                    # Direction Southeast
                    if (int((south_index + distance) / board_size) == int(south_index / board_size)) and (
                            (south_index + distance) < total_cells):
                        southeast_index = south_index + distance
                        if (chess_board[southeast_index] == 1):
                            total_collisions += 1
                            collision_details.append(f'southeast: {index} and {southeast_index}')
                # Check West
                if (int((index - distance) / board_size) == int(index / board_size)) and (index - distance >= 0):
                    west_index = index - distance
                    if (chess_board[west_index] == 1):
                        total_collisions += 1
                        collision_details.append(f'west: {index} and {west_index}')
                # Check East
                if (int((index + distance) / board_size) == int(index / board_size)) and (
                        index + distance < total_cells):
                    east_index = index + distance
                    if (chess_board[east_index] == 1):
                        total_collisions += 1
                        collision_details.append(f'east: {index} and {east_index}')
    return [total_collisions, collision_details]


def find_h_cost(board, n, verbose=False):
    # Function to determine heuristic - total collisions on the board
    collisions, occurrences = find_collisions(board, n)

    if verbose:
        print("Collisions Details:")
        for occ in occurrences:
            print(occ)

    # Return half the collisions, since each colliding position is counted twice from the helper function
    return int(collisions / 2)


def find_best_successor(chessboard, board_size, allow_sideways=False):
    # Function to find the best successor among all possible board configurations
    optimal_successor = []
    current_heuristic = find_h_cost(chessboard, board_size)
    equal_heuristic_successors = []

    for row_index in range(board_size):
        for col_index in range(board_size):
            # Create a temporary board configuration by moving a queen
            temp_board = chessboard[:]
            # Reset the current row
            start_index = row_index * board_size
            temp_board[start_index:start_index + board_size] = [0] * board_size
            # Place the queen in the new column
            temp_board[start_index + col_index] = 1

            temp_heuristic = find_h_cost(temp_board, board_size)
            if allow_sideways:
                if temp_board != chessboard:
                    if temp_heuristic < current_heuristic:
                        optimal_successor = temp_board.copy()
                        current_heuristic = temp_heuristic
                    elif temp_heuristic == current_heuristic:
                        equal_heuristic_successors.append(temp_board)
                        chosen_index = random.randint(0, len(equal_heuristic_successors) - 1)
                        optimal_successor = equal_heuristic_successors[chosen_index]
            else:
                if temp_board != chessboard and temp_heuristic < current_heuristic:
                    optimal_successor = temp_board.copy()
                    current_heuristic = temp_heuristic

    return optimal_successor


def perform_hill_climbing_search(chessboard, board_size, max_tries=1000, show_steps=False):
    # Perform the hill climbing search algorithm, returning the number of steps taken and success status
    num_steps = 0
    is_successful = False
    working_board = chessboard.copy()

    if show_steps:
        print_board(working_board, board_size)

    # Iterate up to the maximum number of tries to find a solution
    for _ in range(max_tries):
        next_board = find_best_successor(working_board, board_size).copy()

        if show_steps and len(next_board) != 0:
            print_board(next_board, board_size)

        num_steps += 1
        # Check if a solution is found
        if len(next_board) != 0 and find_h_cost(next_board, board_size) == 0:
            is_successful = True
            break

        if len(next_board) == 0:  # No more moves available
            break

        working_board = next_board.copy()

    return num_steps, is_successful


# board_size = input_size_board()
total_iterations = 1000 * 100
board_size = 8
total_successes = 0
total_steps_on_success = 0
total_steps_on_failure = 0
total_runtime = 0

print('CS4200 Project 2: N-Queens with n = 8')
print('Hill Climbing Search without sideways:')

for iteration in range(total_iterations):
    start_time = time.time()
    print(f'\nIteration #{iteration + 1}:')
    steps, is_successful = perform_hill_climbing_search(generate_random_board(board_size), board_size, show_steps=True)
    end_time = time.time()

    iteration_runtime = end_time - start_time
    total_runtime += iteration_runtime

    if is_successful:
        print('Success or Solved')
        print(f'Number of steps: {steps}')
        total_successes += 1
        total_steps_on_success += steps
    else:
        print('Failure or Unsolved')
        print(f'Number of steps: {steps}')
        total_steps_on_failure += steps

success_rate = total_successes / total_iterations
failure_rate = 1 - success_rate
average_steps_success = total_steps_on_success / total_successes if total_successes > 0 else 0
average_steps_failure = total_steps_on_failure / (
        total_iterations - total_successes) if total_iterations - total_successes > 0 else 0
average_runtime = total_runtime / total_iterations

print("\n-----REPORT STATISTICS----------")
print(f'N-Queens -> n = {board_size}')
print(f'Number of iterations: {total_iterations}')
print(f'Success Case: {total_successes} / {total_iterations}')
print(f'Success Rate: {success_rate * 100:.2f}%')
# print(f'Failure rate: {failure_rate * 100:.2f}%')
print(f'Average steps on success: {average_steps_success:.2f}')
# print(f'Average steps on failure: {average_steps_failure:.2f}')
print(f'Average runtime per iteration: {average_runtime:.8f} seconds')
print("-----END REPORT STATISTICS-----")
