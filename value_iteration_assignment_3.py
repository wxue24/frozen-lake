import gym
import numpy as np
import collections
import pprint

# from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v1"
GAMMA = 0.9
TEST_EPISODES = 20
SEED = 42

actions = {0: "left", 1: "down", 2: "right", 3: "up"}

initialValues = dict.fromkeys(range(16), 0)

initialPolicy = dict.fromkeys(range(16), "")

# Set all rewards to 0 except goal to equal 1
initialRewards = dict.fromkeys(range(16), 0)
initialRewards[15] = 10

initialTransitions = {}
for s in range(16):
    # states order: left to right, top to bottom
    for a in ["up", "down", "left", "right"]:
        if s in [0, 1, 2, 3] and a == "up":
            # If state is top row, don't include up action
            continue
        if s in [12, 13, 14, 15] and a == "down":
            # If state is bottom row, don't include down action
            continue
        if s in [0, 4, 8, 12] and a == "left":
            # If state is left column, don't include left action
            continue
        if s in [3, 7, 11, 15] and a == "right":
            # If state is right column, don't include right action
            continue

        # determine state after action
        s_a = -1
        if a == "up":
            s_a = s - 4
        elif a == "down":
            s_a = s + 4
        elif a == "left":
            s_a = s - 1
        else:
            s_a = s + 1

        # each (state,action) transitions to a post action state
        initialTransitions[(s, a)] = s_a

# pprint.pprint(initialTransitions)


def printTableFromDict(dict):
    for i in range(16):
        dict[i] = round(dict[i], 2)
    """Prints a table in the shape of a grid from a dictionary where the key is the state"""
    print(f'{dict[0]:5} | {dict[1]:5} | {dict[2]:5} | {dict[3]:5}')
    print(f'{dict[4]:5} | {dict[5]:5} | {dict[6]:5} | {dict[7]:5}')
    print(f'{dict[8]:5} | {dict[9]:5} | {dict[10]:5} | {dict[11]:5}')
    print(f'{dict[12]:5} | {dict[13]:5} | {dict[14]:5} | {dict[15]:5}')
    print("")


class Agent:
    def __init__(self, env):
        self.rewards = initialRewards
        self.values = initialValues
        self.transitions = initialTransitions
        self.policy = initialPolicy
        self.env = env
        self.currentReward = 0
        self.currentRow = 0
        self.currentCol = 0

    @staticmethod
    def create_env():
        env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=True)
        observation, info = env.reset()
        return env

    def update_transits_rewards(self, state, action, new_state, reward):
        pass

    def play_n_random_steps(self, count):
        for i in range(count):
            random_action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(random_action)

            self.currentReward += (GAMMA**i) * reward

            if actions[random_action] == "up":
                self.currentRow -= 1
            if actions[random_action] == "down":
                self.currentRow += 1
            if actions[random_action] == "left":
                self.currentRow -= 1
            if actions[random_action] == "right":
                self.currentRow += 1

            if terminated:
                if reward == 0:
                    self.rewards[obs] = -1
                self.env.reset()
                return

    def print_value_table(self):
        printTableFromDict(self.values)

    def extract_policy(self):
        pass

    def print_policy(self, policy):
        printTableFromDict(policy)

    def calc_action_value(self, state, action):
        pass

    def select_action(self, state):
        pass

    def play_episode(self, env):
        pass

    def value_iteration(self):
        for i in range(100):
            for s in self.values:
                actions_states = []
                for a in ["up", "down", "left", "right"]:
                    if (s, a) in self.transitions:
                        # append (action, transition state) to list
                        actions_states.append((a, self.transitions[(s, a)]))

                probability = 1 / len(actions_states)

                q = -1
                for a, s_a in actions_states:
                    # Apply bellman equation
                    q = max(
                        probability * (self.rewards[s] + GAMMA * self.values[s_a]), q
                    )

                self.values[s] = q


if __name__ == "__main__":
    test_env = Agent.create_env()
    agent = Agent(test_env)

    iter_no = 0
    best_reward = 0
    while True:
        iter_no += 1
        agent.play_n_random_steps(100)
        agent.value_iteration()
        agent.print_value_table()

        reward = agent.currentReward
        if reward > best_reward:
            best_reward = reward
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))

        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            agent.print_value_table()
            policy = agent.extract_policy()
            agent.print_policy(policy)
            break
