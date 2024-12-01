import os
import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import SAC
from tqdm import tqdm

def run_comparison(best_params, num_runs=10, num_episodes=1000):
    configs = {
        'Original': {
            'gamma': 1.0,
            'use_sde': True,
            'train_freq': 32,
        },
        'Best': best_params,
    }

    for config_name, params in configs.items():
        print(f"\nRunning {config_name} Configuration")
        episode_lengths = []
        #episode_lengths = np.zeros((num_runs, num_episodes))

        for run in tqdm(range(num_runs)):
            env = gym.make("MountainCarContinuous-v0")
            monitor_dir = f"comparison_results/{config_name}_run_{run}"
            os.makedirs(monitor_dir, exist_ok=True)
            monitor_file = os.path.join(monitor_dir, "monitor.csv")
            env = Monitor(env, filename=monitor_file)

            # Set default parameters and update with current configuration
            model_params = {
                'policy': "MlpPolicy",
                'env': env,
                'gamma': params.get('gamma', 1.0),
                'learning_rate': params.get('learning_rate', 3e-4),
                'batch_size': params.get('batch_size', 256),
                'tau': params.get('tau', 0.005),
                'use_sde': params.get('use_sde', True),
                'train_freq': params.get('train_freq', 32),
                'verbose': 0,
            }

            # Include optional parameters if provided
            if 'ent_coef' in params:
                model_params['ent_coef'] = params['ent_coef']
            if 'gradient_steps' in params:
                model_params['gradient_steps'] = params['gradient_steps']

            model = SAC(**model_params)
            model.learn(total_timesteps=300000, progress_bar=True)

            # Load episode lengths from monitor.csv
            df = pd.read_csv(monitor_file, comment='#')
            lengths = df['l'].values[:num_episodes]
            episode_lengths.append(lengths)
            #episode_lengths[run, epi]

            env.close()

        # Convert episode_lengths to numpy array
        episode_lengths = np.array(episode_lengths)
        avg_lengths = np.mean(episode_lengths, axis=0)

        # Report average lengths every 10 episodes
        print(f"\Resultados {config_name}:")
        for i in range(0, num_episodes, 10):
            avg_length = np.mean(avg_lengths[:, i:i+10])
            print(f"Episodios {i+1}-{i+10}: Average Length = {avg_length:.2f}")

if __name__ == "__main__":
    # Assuming best_params is obtained from the parameter search
    best_params = {
        # Replace with the actual best parameters from your search
        'learning_rate': 5e-4,
        'batch_size': 256,
        'gamma': 0.99,
        'tau': 0.005,
        'use_sde': True,
        'train_freq': 32,
    }
    run_comparison(best_params)
