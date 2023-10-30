#!/usr/bin/env python3
import gym
from collections import defaultdict, Counter
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v1"
GAMMA = 0.9
ALPHA = 0.2
TEST_EPISODES = 20
SLIPPERY = True


class Agent:
    def __init__(self):
        # Initialize the environment using gym.make with ENV_NAME
        self.env = gym.make("FrozenLake-v1", is_slippery=SLIPPERY)
        # Set the initial state by resetting the environment
        self.env.reset()
        self.state = 0
        # Initialize a default dictionary named values for storing the Q-values
        # Key is the state, action pair
        self.values = defaultdict(float)

    def sample_env(self):
        oldState = self.state
        # Sample a random action from the environment's action space
        action = self.env.action_space.sample()
        # Use the sampled action to take a step in the environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        # If the episode ends, reset the environment and store the new state
        if terminated:
            self.env.reset()
            self.state = 0
        else:
            self.state = obs
        # Return a tuple containing the old state, action, reward, and new state
        return (oldState, action, reward, obs)

    def best_value_and_action(self, state):
        # Initialize variables best_value and best_action to None
        best_value = None
        best_action = None
        # Iterate over all possible actions in the environment's action space
        for action in range(self.env.action_space.n):
            # Calculate the Q-value for each state-action pair
            q = self.values[(state, action)]
            # Update best_value and best_action based on the calculated Q-value
            if best_value == None or q > best_value:
                best_value = q
                best_action = action
        # Return best_value and best_action
        return (best_value, best_action)

    def value_update(self, state, action, reward, new_state):
        # Call the best_value_and_action function to get the best Q-value for the new state
        q_new_state, best_action_new_state = self.best_value_and_action(new_state)
        # Calculate the new Q-value using the reward, gamma, and best Q-value of the new state
        q_state = reward + GAMMA * q_new_state
        # Update the Q-value of the current state-action pair using alpha and the new Q-value
        self.values[(state, action)] = q_state + ALPHA * (
            reward + GAMMA * q_new_state - q_state
        )

    def play_episode(self, env):
        # Initialize a variable total_reward to 0.0
        total_reward = 0.0
        # Reset the environment and store the initial state
        env.reset()
        state = 0
        # Enter a loop that continues until the episode ends
        while True:
            # Call the best_value_and_action function to get the best action for the current state
            best_value, best_action = self.best_value_and_action(state)
            # Take a step in the environment using the best action and store the new state, reward, and done flag
            obs, reward, terminated, truncated, info = env.step(best_action)
            # Update total_reward using the received reward
            total_reward += reward
            # If the episode ends, break from the loop
            if terminated or truncated:
                break
            # Otherwise, update the state using the new state
            else:
                state = obs
        # Return the total reward
        return total_reward

    def print_values(self):
        # Print the Q-values in a readable format
        print("State | Left  | Down  | Right | Up    |")
        for state in range(self.env.observation_space.n):
            print("{0: <5}".format(state), end=" | ")
            for action in range(self.env.action_space.n):
                print(
                    "{0: <5}".format(round(self.values[(state, action)], 3)), end=" | "
                )
            print("")
        print("")
        # Hint: You can use nested loops to iterate over states and actions

    def print_policy(self):
        # Print the policy derived from the Q-values
        # Initialize an empty dictionary named policy
        policy = defaultdict()
        # Iterate over all possible states in the environment
        for s in range(self.env.observation_space.n):
            # Call the best_value_and_action function to get the best action for each state
            best_value, best_action = self.best_value_and_action(s)
            # Update the policy dictionary with the state-action pair
            policy[s] = best_action
            # Print the state and corresponding best action
            print("State: {}, Best Action: {}".format(s, best_action))
        # Return the policy dictionary
        return policy


if __name__ == "__main__":
    test_env = gym.make(ENV_NAME, is_slippery=SLIPPERY)
    agent = Agent()
    writer = SummaryWriter(comment="-q-learning")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        state, action, reward, new_state = agent.sample_env()
        agent.value_update(state, action, reward, new_state)

        cumulative_reward = 0.0
        for _ in range(TEST_EPISODES):
            cumulative_reward += agent.play_episode(test_env)
        cumulative_reward /= TEST_EPISODES
        writer.add_scalar("reward", cumulative_reward, iter_no)
        if cumulative_reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, cumulative_reward))
            best_reward = cumulative_reward
        if cumulative_reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()

    # Print the Q-values and extract/print the policy
    agent.print_values()
    agent.print_policy()
