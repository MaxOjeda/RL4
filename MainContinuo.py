import numpy as np
import gym

from ActorCritic import ActorCritic
from FeatureExtractor import FeatureExtractor
from tqdm import tqdm

def run_actor_critic(runs=30, episodios=1000, intervalo=10):
    episodios_len = np.zeros((runs, episodios))

    for run in tqdm(range(runs)):
        env = gym.make("MountainCarContinuous-v0")
        agent = ActorCritic(alpha_v=0.001, alpha_pi=0.0001, gamma=1.0)

        for episode in range(episodios):
            observation, info = env.reset()
            done = False
            ep_length = 0

            while not done:
                # gaussiana valores
                action, mu, sigma, phi_s = agent.select_action(observation)
                next_observation, reward, terminated, truncated, info = env.step([action])
                done = terminated or truncated
                agent.learn(observation, action, reward, next_observation, done, mu, sigma, phi_s)
                observation = next_observation
                ep_length += 1

            episodios_len[run, episode] = ep_length

        env.close()

    # Promedio largos por run
    promedio_length = np.mean(episodios_len, axis=0)

    # Resultados
    print(f"\Resultados Actor Critic:")
    for i in range(0, episodios, intervalo):
        avg_length = np.mean(promedio_length[i:i+intervalo])
        print(f"Episodios {i+1}-{i+intervalo}: Average Length = {avg_length}")

    return promedio_length

if __name__ == "__main__":
    print("Actor Critic...")
    run_actor_critic()
