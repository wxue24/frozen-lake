import gymnasium as gym

env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=True)
observation, info = env.reset()

for _ in range(1000):
    action = (
        env.action_space.sample()
    )  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if reward == 1:
        print(observation, reward)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
