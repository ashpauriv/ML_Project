# models/q_learning.py
import numpy as np

class QLearning:
    def __init__(self, states, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = np.zeros((len(states), len(actions)))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.states = states
        self.actions = actions

    def choose_action(self, state_index):
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(self.actions))
        return np.argmax(self.q_table[state_index])

    def update(self, state_index, action_index, reward, next_state_index):
        current_q = self.q_table[state_index, action_index]
        max_future_q = np.max(self.q_table[next_state_index])
        new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * max_future_q)
        self.q_table[state_index, action_index] = new_q


def run_q_learning():
    print("Running Q-Learning Simulation...")

    # Define states and actions
    states = [(demand, supply) for demand in range(1, 5) for supply in range(1, 5)]
    actions = [10, 15, 20, 25]  # Example price actions

    # Initialize Q-Learning with states and actions
    ql_model = QLearning(states, actions, alpha=0.1, gamma=0.9, epsilon=0.1)

    # Simulate Q-learning updates
    for episode in range(1000):  # Number of episodes
        state_index = np.random.choice(len(states))  # Randomly start in a state
        while True:
            action_index = ql_model.choose_action(state_index)
            price = actions[action_index]

            # Define a reward function for illustration
            demand, supply = states[state_index]
            reward = demand * 5 - price  # Simplified reward function

            # Transition to the next state (randomly for this example)
            next_state_index = np.random.choice(len(states))

            # Update Q-table based on the reward received
            ql_model.update(state_index, action_index, reward, next_state_index)

            # End the episode randomly to simulate dynamic transitions
            if np.random.rand() < 0.1:
                break

            # Move to the next state
            state_index = next_state_index

    print("Q-Learning completed. Q-table:")
    print(ql_model.q_table)

    # Return the Q-table for display in main_script
    return ql_model.q_table
