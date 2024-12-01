import gymnasium as gym
import numpy as np

from Algoritmos import Sarsa
from Algoritmos import QLearning
from Algoritmos import SarsaLambda
from tqdm import tqdm

def experiment(agent_class, runs=30, episodios=1000, intervalo=10, **agent_kwargs):
    ep_length = np.zeros((runs, episodios))

    for run in tqdm(range(runs)):
        env = gym.make("MountainCar-v0")
        n_action = env.action_space.n

        agent = agent_class(n_action, **agent_kwargs)
        for episode in range(episodios):
            observation, info = env.reset()
            action = agent.sample_action(observation)
            done = truncado = False
            ep_length = 0

            while not done and not truncado:
                next_observation, reward, done, truncado, info = env.step(action)
                ep_length += 1

                # Cabmiar dependiendo si son Sarsa o Q
                if isinstance(agent, (Sarsa, SarsaLambda)):
                    next_action = agent.sample_action(next_observation)
                    agent.learn(observation, action, reward, next_observation, next_action, done)
                    observation, action = next_observation, next_action
                elif isinstance(agent, QLearning):
                    agent.learn(observation, action, reward, next_observation, done)
                    observation = next_observation
                    action = agent.sample_action(observation)
            ep_length[run, episode] = ep_length

        env.close()

    promedio_length = np.mean(ep_length, axis=0)

    # Mostrar resultados
    print(f"\Resultados {agent_class.__name__}:")
    for i in range(0, episodios, intervalo):
        avg_length = np.mean(promedio_length[i:i+intervalo])
        print(f"Espisodios {i+1}-{i+intervalo}: Average Length = {avg_length}")

    return promedio_length

if __name__ == "__main__":
    runs = 30
    episodios = 1000
    intervalo = 10

    # Parametros
    gamma = 1.0
    epsilon = 0.0
    alpha = 0.5 / 8
    lambd = 0.5

    # Ejecutar cada experimento
    print("Sarsa...")
    sarsa_lengths = experiment(Sarsa, runs, episodios, intervalo, epsilon=epsilon, alpha=alpha, gamma=gamma)

    print("\nQ-Learning...")
    qlearning_lengths = experiment(QLearning, runs, episodios, intervalo, epsilon=epsilon, alpha=alpha, gamma=gamma)

    print("\nSarsa(l)...")
    sarsa_lambda_lengths = experiment(SarsaLambda, runs, episodios, intervalo, epsilon=epsilon, alpha=alpha, gamma=gamma, lambd=lambd)
