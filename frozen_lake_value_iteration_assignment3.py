import gym
from collections import defaultdict

ENV_NAME = "FrozenLake-v1"
GAMMA = 0.9
TEST_EPISODES = 20
SEED = 42
SLIPPERY = False

actions = {0: "left", 1: "down", 2: "right", 3: "up"}

initialTransitions = {}
for s in range(16):
    # states order: left to right, top to bottom
    for a in [0, 1, 2, 3]:
        # determine state after action
        s_a = -1
        if a == 0:
            s_a = s - 1
        elif a == 1:
            s_a = s + 4
        elif a == 2:
            s_a = s + 1
        elif a == 3:
            s_a = s - 4

        # if state is out of bounds, remain unchanged
        if s_a < 0 or s_a > 15:
            s_a = s

        # each (state,action) transitions to a post action state
        d = defaultdict(int)
        d[s_a] = 1
        initialTransitions[(s, a)] = d


class Agent:
    def __init__(self):
        # Define env
        self.env = self.create_env()
        # Define state
        self.state = 0
        # Define rewards
        self.rewards = defaultdict(int)
        # Define transits
        self.transits = initialTransitions
        # Define values
        self.values = defaultdict(float)

    @staticmethod
    def create_env():
        env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=SLIPPERY)
        env.reset()
        return env

    def update_transits_rewards(self, state, action, new_state, reward):
        # Get the key, which is a state action pair
        key = (state, action)
        # update rewards which is accessed by key plus the new state
        self.rewards[(key, new_state)] = reward
        # update transits count which is accessed by key and new_state
        self.transits[key][new_state] += 1

    def play_n_random_steps(self, count):
        # for loop that iterates count number of times
        for i in range(count):
            # get an action
            random_action = self.env.action_space.sample()
            # step through the environment
            obs, reward, terminated, truncated, info = self.env.step(random_action)
            # update the transits rewards
            self.update_transits_rewards(self.state, random_action, obs, reward)
            # update the state
            self.state = obs
            if terminated:
                self.env.reset()
                self.state = 0

    def print_value_table(self):
        for i in range(4):
            for j in range(4):
                print(round(self.values[4 * i + j], 3), end=" | ")
            print("")
        print("")

    def extract_policy(self):
        # Define policy as an empty list
        policy = []
        # for every state
        for s in range(16):
            # select the action
            best_action = self.select_action(s)
            # append action to the policy
            policy.append(best_action)
        # return policy
        return policy

    def print_policy(self, policy):
        # define actions in NL
        # nested for loop to print the actions in 2d matrix format
        for i in range(4):
            for j in range(4):
                print(actions[policy[4 * i + j]], end=" | ")
            print("")
        print("")

    def calc_action_value(self, state, action):
        # get target counts which access transits by state, action
        counts = self.transits[(state, action)]
        # get the sum of all the counts
        total_count = 0
        for s_a in counts:
            total_count += counts[s_a]
        # for each target state
        sum = 0
        for s_a in counts:
            # calculate the proportion of reward plus gamma * value of the target state, then sum it all together.
            proportion = counts[s_a] / total_count
            sum += proportion + GAMMA * self.values[s_a]
        # return that sum
        # print((state, actions[action], counts, sum))

        return sum

    def select_action(self, state):
        # define best action and best value
        best_action = 0
        best_value = 0
        # For action in the range of actions
        for action in range(4):
            # calculate the action value
            value = self.calc_action_value(state, action)
            print(actions[action], value)
            # if best value is less than action value
            if best_value < value:
                # update best value and best action
                best_action = action
                best_value = value
        # return best action
        return best_action

    def play_episode(self, env):
        env.reset()
        # define reward and state
        reward = 0
        state = 0
        # While loop
        while True:
            # select an action
            action = self.select_action(state)
            print(actions[action])
            # take a step
            obs, r, terminated, truncated, info = env.step(action)
            
            # state = obs
            # reward += r
            # if terminated:
            #     return reward
            # if state is multiple
            
            # update reward
            # update count
            # else
            # update reward
            # update count
            # update total reward
            # get out if we're done
            # set state to new state
            # return total reward

    def value_iteration(self):
        # for each state
        for s in range(16):
            # set state_values equalt to a list of calc_action_value for every action
            state_values = [self.calc_action_value(s, a) for a in range(4)]
            # set self values to the max state_values
            self.values[s] = max(state_values)


if __name__ == "__main__":
    test_env = Agent.create_env()
    agent = Agent()

    iter_no = 0
    best_reward = 0
    while True:
        iter_no += 1
        print("Iteration {}".format(iter_no))
        agent.play_n_random_steps(20)
        agent.value_iteration()
        agent.print_value_table()

        reward = 0  # sum of play episode for all 20 episodes / number of episodes
        for i in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
            print("Test Episode {}: Reward {}".format(i, reward))
        reward /= TEST_EPISODES

        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))

        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            agent.print_value_table()
            policy = agent.extract_policy()
            agent.print_policy(policy)
            break
