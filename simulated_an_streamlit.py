import streamlit as st
import numpy as np
import random
import matplotlib.pyplot as plt


# -----------------------------
# 1. Create the Dataset
# -----------------------------
def create_knapsack_dataset(number_of_items, max_utility=100, max_weight=100):
    """
    Randomly generates utilities and weights for a given number of items.
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
    The solution is a binary vector indicating which items are selected.
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
    Solves the knapsack problem using the simulated annealing algorithm.

    Parameters:
      utilities, weights : Arrays with the values and weights of the items.
      max_capacity       : Maximum allowed knapsack capacity.
      initial_solution   : Initial binary vector representing the chosen items.
      T0                 : Initial temperature.
      alpha              : Cooling factor (for exponential cooling, T = T * alpha; for linear, T = T - alpha).
      iterations         : Total number of iterations.
      cooling_method     : "exponential" or "linear".

    Returns:
      best_solution, best_obj, fig : The best solution found, its objective value, and a figure with the plots.
    """
    # Initialize the current and best solutions using the initial solution.
    current_solution = initial_solution.copy()
    best_solution = current_solution.copy()
    current_obj = objective_function(utilities, weights, max_capacity, current_solution)
    best_obj = current_obj

    T = T0  # Set the initial temperature
    T_list = []  # To record the evolution of temperature
    obj_list = []  # To record the evolution of the best objective value

    for i in range(iterations):
        # --- Generate a neighbor ---
        neighbor = current_solution.copy()
        index = random.randint(0, len(neighbor) - 1)
        neighbor[index] = 1 - neighbor[index]  # Flip the decision (0 -> 1 or 1 -> 0)

        # Evaluate the neighbor solution.
        neighbor_obj = objective_function(utilities, weights, max_capacity, neighbor)
        delta = neighbor_obj - current_obj

        # --- Neighbor Acceptance Decision ---
        if delta > 0:
            current_solution = neighbor
            current_obj = neighbor_obj
        else:
            acceptance_probability = np.exp(delta / T) if T > 0 else 0
            if random.random() < acceptance_probability:
                current_solution = neighbor
                current_obj = neighbor_obj

        # Update the best solution, if applicable.
        if current_obj > best_obj:
            best_solution = current_solution.copy()
            best_obj = current_obj

        # Store the values for plotting.
        T_list.append(T)
        obj_list.append(best_obj)

        # --- Update the Temperature ---
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

    # Create the plots using matplotlib.
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
# 4. Streamlit Interface
# -----------------------------
st.title("Knapsack Problem with Simulated Annealing")
st.markdown(
    """
This application uses the simulated annealing algorithm to solve the knapsack problem.
Adjust the parameters in the sidebar and click **"Run Simulated Annealing"** to execute the algorithm.
"""
)

# Problem configuration in the sidebar
st.sidebar.header("Problem Settings")
number_of_items = st.sidebar.number_input(
    "Number of Items", min_value=1, value=100, step=1
)
max_utility = st.sidebar.number_input("Maximum Utility", min_value=1, value=100, step=1)
max_weight = st.sidebar.number_input("Maximum Weight", min_value=1, value=100, step=1)
max_capacity = st.sidebar.number_input(
    "Knapsack Maximum Capacity", min_value=1, value=2500, step=1
)

st.sidebar.header("Algorithm Settings")
T0 = st.sidebar.number_input(
    "Initial Temperature (T0)", min_value=0.0, value=1000.0, step=1.0, format="%.2f"
)
alpha = st.sidebar.number_input(
    "Cooling Rate (alpha)",
    min_value=0.0,
    max_value=1.0,
    value=0.995,
    step=0.001,
    format="%.3f",
)
iterations = st.sidebar.number_input(
    "Number of Iterations", min_value=1, value=1000, step=1
)
cooling_method = st.sidebar.selectbox(
    "Cooling Method", options=["exponential", "linear"]
)

# We remove the seed setting to ensure each execution is random.
# To use reproducibility, uncomment the lines below and set a seed:
# seed = st.sidebar.number_input("Random Seed", min_value=0, value=42, step=1)
# np.random.seed(seed)
# random.seed(seed)

# Button to run the algorithm
if st.button("Run Simulated Annealing"):
    st.write("### Executing the algorithm...")

    # Generate the item dataset.
    utilities, weights = create_knapsack_dataset(
        number_of_items, max_utility, max_weight
    )

    # Generate an initial random solution (binary vector).
    initial_solution = np.random.choice([0, 1], size=number_of_items)

    # Display the initial solution.
    st.write("### Initial Solution")
    st.write(initial_solution)

    # Run the simulated annealing algorithm.
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

    # Calculate the total weight of the best solution.
    total_weight = np.dot(weights, best_solution)

    # Display numerical results.
    st.write("### Numerical Results")
    st.write(f"**Best Objective Value Found:** {best_obj}")
    st.write(f"**Total Weight of the Best Solution:** {total_weight}")

    # Display the plots.
    st.write("### Parameter Evolution")
    st.pyplot(fig)
