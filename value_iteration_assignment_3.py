import gym
import numpy as np
import collections
import pprint

# from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v1"
GAMMA = 0.9
TEST_EPISODES = 20
SEED = 43
REWARD_HOLE = -5
REWARD_GOAL = 5
REWARD_MOVE = 0

actions = {0: "left", 1: "down", 2: "right", 3: "up"}

initialValues = dict.fromkeys(range(16), 0)


# Set all rewards to 0 except goal to equal 1
initialRewards = dict.fromkeys(range(16), REWARD_MOVE)
initialRewards[15] = REWARD_GOAL

initialTransitions = {}
for s in range(16):
    # states order: left to right, top to bottom
    for a in ["up", "down", "left", "right"]:
        # if s in [0, 1, 2, 3] and a == "up":
        #     # If state is top row, don't include up action
        #     continue
        # if s in [12, 13, 14, 15] and a == "down":
        #     # If state is bottom row, don't include down action
        #     continue
        # if s in [0, 4, 8, 12] and a == "left":
        #     # If state is left column, don't include left action
        #     continue
        # if s in [3, 7, 11, 15] and a == "right":
        #     # If state is right column, don't include right action
        #     continue

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
        if s_a < 0 or s_a > 15:
            s_a = s

        # each (state,action) transitions to a post action state
        d = collections.defaultdict(int)
        d[s_a] = 1
        initialTransitions[(s, a)] = d

# pprint.pprint(initialTransitions)


def printTableFromDict(dict):
    for i in range(16):
        dict[i] = round(dict[i], 2)
    """Prints a table in the shape of a grid from a dictionary where the key is the state"""
    print(f"{dict[0]:5} | {dict[1]:5} | {dict[2]:5} | {dict[3]:5}")
    print(f"{dict[4]:5} | {dict[5]:5} | {dict[6]:5} | {dict[7]:5}")
    print(f"{dict[8]:5} | {dict[9]:5} | {dict[10]:5} | {dict[11]:5}")
    print(f"{dict[12]:5} | {dict[13]:5} | {dict[14]:5} | {dict[15]:5}")
    print("")


class Agent:
    def __init__(self, env, slippery):
        self.rewards = initialRewards
        self.values = initialValues
        self.transitions = initialTransitions
        self.env = env
        self.slippery = slippery

    @staticmethod
    def create_env(slippery):
        env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=slippery)
        observation, info = env.reset(seed=SEED)
        return env

    def update_transits_rewards(self, state, action, new_state, reward):
        """
        For when agent slips into an unintended state (slippery = True)
        """
        print(
            "({},{}), Old state {}, new state {}".format(
                state, action, self.transitions[(state, action)], new_state
            )
        )
        self.transitions[(state, action)] = new_state

    def play_n_random_steps(self, count):
        """Play n random steps"""
        state = 0
        for i in range(count):
            random_action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(random_action)

            self.transitions[(state, actions[random_action])][obs] += 1
            state = obs

            if terminated:
                # If agent falls into hole, update reward at that state
                if reward == 0:
                    self.rewards[obs] = REWARD_HOLE
                state = 0
                self.env.reset(seed=SEED)

        # pprint.pprint(self.transitions)

    def print_value_table(self):
        """Prints a table in the shape of a grid from a dictionary where the key is the state"""
        for i in range(16):
            self.values[i] = round(self.values[i], 2)
        print(
            f"{self.values[0]:5} | {self.values[1]:5} | {self.values[2]:5} | {self.values[3]:5}"
        )
        print(
            f"{self.values[4]:5} | {self.values[5]:5} | {self.values[6]:5} | {self.values[7]:5}"
        )
        print(
            f"{self.values[8]:5} | {self.values[9]:5} | {self.values[10]:5} | {self.values[11]:5}"
        )
        print(
            f"{self.values[12]:5} | {self.values[13]:5} | {self.values[14]:5} | {self.values[15]:5}"
        )
        print("")

    def extract_policy(self):
        """Extracts policy at each state and returns as a dict"""
        policy = dict.fromkeys(range(16), "")
        for s in range(16):
            action, actionNum, value = self.select_action(s)
            policy[s] = action
        return policy

    def print_policy(self, policy):
        """Prints policy as a table"""
        print(f"{policy[0]:5} | {policy[1]:5} | {policy[2]:5} | {policy[3]:5}")
        print(f"{policy[4]:5} | {policy[5]:5} | {policy[6]:5} | {policy[7]:5}")
        print(f"{policy[8]:5} | {policy[9]:5} | {policy[10]:5} | {policy[11]:5}")
        print(f"{policy[12]:5} | {policy[13]:5} | {policy[14]:5} | {policy[15]:5}")
        print("")

    def select_action(self, state):
        "Returns tuple of best action and value"
        actions_values = []
        # determine which actions can be taken and append the action-value pair
        for a in ["up", "down", "left", "right"]:
            s_a = -1
            if a == "up":
                s_a = state - 4
            elif a == "down":
                s_a = state + 4
            elif a == "left":
                s_a = state - 1
            else:
                s_a = state + 1

            if s_a > -1 and s_a < 16:
                actions_values.append((a, self.values[s_a]))

        bestAction = None
        bestValue = 0
        actionNum = 0
        for action, value in actions_values:
            if value > bestValue:
                bestAction = action
                bestValue = value

        if bestAction == "left":
            actionNum = 0
        if bestAction == "down":
            actionNum = 1
        if bestAction == "right":
            actionNum = 2
        if bestAction == "up":
            actionNum = 3

        # print(bestAction, actionNum, bestValue)
        return (bestAction, actionNum, bestValue)

    def value_iteration(self):
        """Performs value iteration"""
        for i in range(100):
            for s in self.values:
                # list of values for each action
                v = []

                for a in ["up", "down", "left", "right"]:
                    # total count
                    total_count = 0

                    for s_a, count in self.transitions[(s, a)].items():
                        # print(self.transitions[(s, a)][s_a])
                        total_count += count

                    # calculate q
                    if total_count > 0:
                        q = 0
                        for s_a, count in self.transitions[(s, a)].items():
                            prob = count / total_count
                            q += prob * (self.rewards[s] + GAMMA * self.values[s_a])
                        v.append(q)
                # Get maximizing action
                self.values[s] = max(v)
                

    def play_episode(self):
        """Play episode using policy, returns whether the agent finished or not"""
        self.env.reset(seed=SEED)
        state = 0
        print("")
        while True:
            bestAction, actionNum, bestValue = agent.select_action(state)
            obs, reward, terminated, truncated, info = self.env.step(actionNum)
            print("{} -> {} -> {}".format(state, bestAction, obs))

            if terminated:
                if reward == 1:
                    return True
                else:
                    return False

            state = obs


if __name__ == "__main__":
    slippery = True
    test_env = Agent.create_env(slippery)
    agent = Agent(test_env, slippery)

    iter_no = 1
    best_reward = 0
    while True:
        print("Iteration {}".format(iter_no))
        print("Playing random steps ...")
        iter_no += 1
        agent.play_n_random_steps(100)
        agent.value_iteration()

        # Testing
        agent.print_value_table()
        policy = agent.extract_policy()
        agent.print_policy(policy)

        # Run 20 episodes here to calculate reward
        print("running test episodes ...")
        reward = 0
        for i in range(TEST_EPISODES):
            print("Playing episode {}".format(i))
            # policy = agent.extract_policy()
            # print("Policy")
            # print(agent.print_policy(policy))
            if agent.play_episode() == True:
                reward += 1
        reward /= 20

        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward
        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            print("Value table")
            agent.print_value_table()
            policy = agent.extract_policy()
            print("Policy table")
            agent.print_policy(policy)
            break
