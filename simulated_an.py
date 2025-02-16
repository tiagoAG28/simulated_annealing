# Tópico 2 - Problema da Mochila (Knapsack Problem) - Simulated Annealing
# Tiago Gomes


import numpy as np
import random
import matplotlib.pyplot as plt


# -----------------------------
# 1. Create the Dataset
# -----------------------------
def create_knapsack_dataset(number_of_items, max_utility=100, max_weight=100):
    """
    Generates random utilities and weights for a given number of items.
    Each utility and weight is an integer between 1 and the provided maximum value.
    """
    utilities = np.random.randint(1, max_utility, number_of_items)
    weights = np.random.randint(1, max_weight, number_of_items)
    return utilities, weights


# -----------------------------
# 2. Objective Function
# -----------------------------
def objective_function(utilities, weights, max_capacity, solution):
    """
    Computes the total utility of a solution.
    The solution is a binary vector that indicates which items are selected.
    If the total weight exceeds the knapsack capacity, returns 0 (penalizes infeasible solutions).
    """
    total_weight = np.dot(weights, solution)
    if total_weight <= max_capacity:
        return np.dot(utilities, solution)
    else:
        return 0  # Infeasible solution receives a score of zero


# -----------------------------
# 3. Simulated Annealing Algorithm
# -----------------------------
def simulated_annealing(
    utilities,
    weights,
    max_capacity,
    initial_solution,
    T0,
    alpha,
    iterations,
    cooling_method="exponential",
):
    """
    Solves the knapsack problem using simulated annealing.

    Parameters:
      utilities, weights : Arrays with item values and weights.
      max_capacity       : Maximum allowed knapsack capacity.
      initial_solution   : Initial binary vector representing the selected items.
      T0                 : Initial temperature.
      alpha              : Cooling factor (for exponential cooling, T = T * alpha; for linear, T = T - alpha).
      iterations         : Total number of iterations.
      cooling_method     : "exponential" or "linear".

    Returns:
      best_solution, best_obj, fig : The best solution found, its objective value, and a figure with the plots.
    """
    # 1. Initialization of variables and solutions
    current_solution = initial_solution.copy()
    best_solution = current_solution.copy()
    current_obj = objective_function(utilities, weights, max_capacity, current_solution)
    best_obj = current_obj

    T = T0  # Set the initial temperature
    T_list = []  # List to record temperature evolution
    obj_list = []  # List to record the evolution of the best objective value

    # 2. Main loop of iterations
    for i in range(iterations):
        # 3. Generation of a neighbor
        neighbor = current_solution.copy()
        index = random.randint(0, len(neighbor) - 1)
        neighbor[index] = 1 - neighbor[index]  # Flip the decision (0 -> 1 or 1 -> 0)

        # 4. Evaluation of the neighbor solution and calculation of delta
        neighbor_obj = objective_function(utilities, weights, max_capacity, neighbor)
        delta = neighbor_obj - current_obj

        # 5. Acceptance criteria for the neighbor
        if delta > 0:
            current_solution = neighbor
            current_obj = neighbor_obj
        else:
            acceptance_probability = np.exp(delta / T) if T > 0 else 0
            if random.random() < acceptance_probability:
                current_solution = neighbor
                current_obj = neighbor_obj

        # 6. Updating the best solution found
        if current_obj > best_obj:
            best_solution = current_solution.copy()
            best_obj = current_obj

        # 7. Storing values for plotting
        T_list.append(T)
        obj_list.append(best_obj)

        # 8. Updating the temperature
        if cooling_method == "exponential":
            T = T * alpha  # Exponential decay
        elif cooling_method == "linear":
            T = T - alpha  # Here, 'alpha' is the fixed decrement
            if T < 0:
                T = 0
        else:
            raise ValueError(
                "Cooling method not recognized. Use 'exponential' or 'linear'."
            )

    # 9. Creating the plots and returning the results
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(T_list, color="blue")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Temperature")
    ax[0].set_title("Temperature Evolution")

    ax[1].plot(obj_list, color="green")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Best Objective Value")
    ax[1].set_title("Best Objective Evolution")

    plt.tight_layout()

    return best_solution, best_obj, fig


# -----------------------------
# 4. Main Function and Execution
# -----------------------------
def main():
    print("Knapsack Problem with Simulated Annealing")
    print("=========================================")

    # Request parameters from the user
    try:
        number_of_items = int(input("Enter the number of items (e.g., 100): "))
    except:
        number_of_items = 100
    try:
        max_utility = int(input("Enter the maximum utility per item (e.g., 100): "))
    except:
        max_utility = 100
    try:
        max_weight = int(input("Enter the maximum weight per item (e.g., 100): "))
    except:
        max_weight = 100
    try:
        max_capacity = int(
            input("Enter the knapsack's maximum capacity (e.g., 2500): ")
        )
    except:
        max_capacity = 2500

    try:
        T0 = float(input("Enter the initial temperature (T0) (e.g., 1000): "))
    except:
        T0 = 1000.0
    try:
        alpha = float(input("Enter the cooling rate (alpha) (e.g., 0.995): "))
    except:
        alpha = 0.995
    try:
        iterations = int(input("Enter the number of iterations (e.g., 1000): "))
    except:
        iterations = 1000

    cooling_method = input("Enter the cooling method ('exponential' or 'linear'): ")
    if cooling_method not in ["exponential", "linear"]:
        cooling_method = "exponential"

    try:
        seed_val = input("Enter a random seed (optional, press Enter to skip): ")
        if seed_val != "":
            seed_val = int(seed_val)
            np.random.seed(seed_val)
            random.seed(seed_val)
    except:
        pass

    # Generate the dataset
    utilities, weights = create_knapsack_dataset(
        number_of_items, max_utility, max_weight
    )

    # Generate the initial solution (binary vector)
    initial_solution = np.random.choice([0, 1], size=number_of_items)
    print("\nInitial Solution:")
    print(initial_solution)

    # Execute the algorithm
    best_solution, best_obj, fig = simulated_annealing(
        utilities,
        weights,
        max_capacity,
        initial_solution,
        T0,
        alpha,
        iterations,
        cooling_method=cooling_method,
    )

    total_weight = np.dot(weights, best_solution)

    print("\nNumerical Results:")
    print("Best Objective Value Found:", best_obj)
    print("Total Weight of the Best Solution:", total_weight)

    # Display the plots
    fig.show()  # Opens the window with the plots
    plt.show()  # Ensures the plots are displayed


if __name__ == "__main__":
    main()
