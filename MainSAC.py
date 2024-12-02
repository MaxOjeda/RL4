import os
import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

from stable_baselines3 import SAC
# from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm

def run_sac(num_runs=30, episodios=1000):
    episode_lengths = []

    for run in tqdm(range(num_runs)):
        env = gym.make("MountainCarContinuous-v0")
        # Archivo Monitor
        monitor_dir = f"Results/results_sac_run_{run}"
        os.makedirs(monitor_dir, exist_ok=True)
        monitor_file = os.path.join(monitor_dir, "monitor.csv")

        env = Monitor(env, filename=monitor_file)
        model = SAC(
            "MlpPolicy",
            env,
            gamma=1.0,
            use_sde=True,
            train_freq=32,
            verbose=0
        )
        model.learn(total_timesteps=300000, progress_bar=True)

        # Cargar monitor.csv
        df = pd.read_csv(os.path.join(monitor_dir, "monitor.csv"), skiprows=1)
        lengths = df['l'].values[:episodios]
        episode_lengths.append(lengths)

        env.close()

    # Proemdio length
    episode_lengths = np.array(episode_lengths)
    avg_lengths = np.mean(episode_lengths, axis=0)

    # Resultados cada 10 episodios
    print(f"\nResultados SAC:")
    for i in range(0, episodios, 10):
        avg_length = np.mean(avg_lengths[i:i+10])
        print(f"Episodios {i+1}-{i+10}: Average Length = {avg_length}")

    return avg_lengths

if __name__ == "__main__":
    print("\nSAC...")
    run_sac()
