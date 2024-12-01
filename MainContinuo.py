import numpy as np
import gym

from ActorCritic import ActorCritic
from FeatureExtractor import FeatureExtractor
from tqdm import tqdm

def run_actor_critic(num_runs=30, num_episodes=1000, report_interval=10):
    episode_lengths = np.zeros((num_runs, num_episodes))

    for run in tqdm(range(num_runs)):
        env = gym.make("MountainCarContinuous-v0")
        agent = ActorCritic(alpha_v=0.001, alpha_pi=0.0001, gamma=1.0)

        for episode in range(num_episodes):
            observation, info = env.reset()
            done = False
            ep_length = 0

            while not done:
                action, mu, sigma, phi_s = agent.select_action(observation)
                next_observation, reward, terminated, truncated, info = env.step([action])
                done = terminated or truncated
                agent.learn(observation, action, reward, next_observation, done, mu, sigma, phi_s)
                observation = next_observation
                ep_length += 1

            episode_lengths[run, episode] = ep_length

        env.close()

    # Compute average episode lengths over runs
    avg_lengths = np.mean(episode_lengths, axis=0)

    # Report average lengths every 'report_interval' episodes
    print(f"\nResults for ActorCritic:")
    for i in range(0, num_episodes, report_interval):
        avg_length = np.mean(avg_lengths[i:i+report_interval])
        print(f"Episodes {i+1}-{i+report_interval}: Average Length = {avg_length}")

    return avg_lengths

if __name__ == "__main__":
    print("Running ActorCritic...")
    run_actor_critic(num_runs=20)
