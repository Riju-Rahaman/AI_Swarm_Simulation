import numpy as np
import matplotlib.pyplot as plt
import time

# Grid size
GRID_SIZE = 10

# Rewards and penalties
GOAL_REWARD = 100
OBSTACLE_PENALTY = -10
STEP_COST = -1

# Actions (up, down, left, right)
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
ACTION_TO_DELTA = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}

# Create the environment
def create_environment(grid_size):
    grid = np.zeros((grid_size, grid_size))
    grid[grid_size - 1, grid_size - 1] = GOAL_REWARD  # Goal position
    obstacles = [(2, 2), (3, 3), (4, 4), (5, 5)]  # Obstacle positions
    for obs in obstacles:
        grid[obs] = OBSTACLE_PENALTY
    return grid

# Initialize Q-table
def initialize_q_table(grid_size):
    return np.zeros((grid_size, grid_size, len(ACTIONS)))

# Choose action using epsilon-greedy policy
def choose_action(state, q_table, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(ACTIONS)
    else:
        return ACTIONS[np.argmax(q_table[state[0], state[1]])]

# Take action and get the next state and reward
def take_action(state, action, grid):
    delta = ACTION_TO_DELTA[action]
    next_state = (state[0] + delta[0], state[1] + delta[1])

    # Check boundaries
    if not (0 <= next_state[0] < grid.shape[0] and 0 <= next_state[1] < grid.shape[1]):
        next_state = state

    reward = grid[next_state]
    return next_state, reward

# Q-learning algorithm
def q_learning(grid, episodes, alpha, gamma, epsilon):
    q_table = initialize_q_table(GRID_SIZE)
    for episode in range(episodes):
        state = (0, 0)  # Starting position
        while state != (GRID_SIZE - 1, GRID_SIZE - 1):  # Until the goal is reached
            action = choose_action(state, q_table, epsilon)
            next_state, reward = take_action(state, action, grid)
            action_index = ACTIONS.index(action)

            # Update Q-value
            q_table[state[0], state[1], action_index] += alpha * (
                reward + gamma * np.max(q_table[next_state[0], next_state[1]]) -
                q_table[state[0], state[1], action_index]
            )

            state = next_state
    return q_table

# Visualize the grid and agent movement
def visualize(grid, path):
    plt.ion()
    fig, ax = plt.subplots()
    for step, state in enumerate(path):
        grid_copy = grid.copy()
        grid_copy[state] = 50  # Highlight the agent's current position
        ax.clear()
        ax.imshow(grid_copy, cmap="coolwarm", interpolation="none")
        ax.set_title(f"Step {step + 1}")
        plt.pause(0.5)
    plt.ioff()
    plt.show()

# Main function
def main():
    grid = create_environment(GRID_SIZE)
    q_table = q_learning(grid, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1)

    # Test the trained agent
    state = (0, 0)
    path = [state]
    while state != (GRID_SIZE - 1, GRID_SIZE - 1):
        action = ACTIONS[np.argmax(q_table[state[0], state[1]])]
        state, _ = take_action(state, action, grid)
        path.append(state)

    print("Agent path:", path)

    # Visualize the agent's movement
    visualize(grid, path)

if __name__ == "__main__":
    main()
