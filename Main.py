import gymnasium as gym
import numpy as np

from Algoritmos import Sarsa
from Algoritmos import QLearning
from Algoritmos import SarsaLambda
from tqdm import tqdm

def run_experiment(agent_class, num_runs=30, num_episodes=1000, report_interval=10, **agent_kwargs):
    episode_lengths = np.zeros((num_runs, num_episodes))

    for run in tqdm(range(num_runs)):
        env = gym.make("MountainCar-v0")
        n_actions = env.action_space.n

        agent = agent_class(n_actions, **agent_kwargs)
        for episode in range(num_episodes):
            observation, info = env.reset()
            action = agent.sample_action(observation)
            terminated = truncated = False
            ep_length = 0

            while not terminated and not truncated:
                next_observation, reward, terminated, truncated, info = env.step(action)
                ep_length += 1

                if isinstance(agent, (Sarsa, SarsaLambda)):
                    next_action = agent.sample_action(next_observation)
                    agent.learn(observation, action, reward, next_observation, next_action, terminated)
                    observation, action = next_observation, next_action
                elif isinstance(agent, QLearning):
                    agent.learn(observation, action, reward, next_observation, terminated)
                    observation = next_observation
                    action = agent.sample_action(observation)
            episode_lengths[run, episode] = ep_length

        env.close()

    # Compute average episode lengths over runs
    avg_lengths = np.mean(episode_lengths, axis=0)

    # Report average lengths every 'report_interval' episodes
    print(f"\Resultados {agent_class.__name__}:")
    for i in range(0, num_episodes, report_interval):
        avg_length = np.mean(avg_lengths[i:i+report_interval])
        print(f"Espisodios {i+1}-{i+report_interval}: Average Length = {avg_length}")

    return avg_lengths

if __name__ == "__main__":
    num_runs = 30
    num_episodes = 1000
    report_interval = 10

    # Parameters
    gamma = 1.0
    epsilon = 0.0
    alpha = 0.5 / 8
    lambd = 0.5

    print("Sarsa...")
    sarsa_lengths = run_experiment(
        Sarsa,
        num_runs,
        num_episodes,
        report_interval,
        epsilon=epsilon,
        alpha=alpha,
        gamma=gamma,
    )

    print("\nQ-Learning...")
    qlearning_lengths = run_experiment(
        QLearning,
        num_runs,
        num_episodes,
        report_interval,
        epsilon=epsilon,
        alpha=alpha,
        gamma=gamma,
    )

    print("\nSarsa(l)...")
    sarsa_lambda_lengths = run_experiment(
        SarsaLambda,
        num_runs,
        num_episodes,
        report_interval,
        epsilon=epsilon,
        alpha=alpha,
        gamma=gamma,
        lambd=lambd,
    )
