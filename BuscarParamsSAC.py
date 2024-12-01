import os
import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import SAC
from tqdm import tqdm

def run_sac_parameter_search(num_configs=10, num_episodes=1000):
    # Define a list of parameter configurations to test
    parameter_configs = [
        {'learning_rate': 1e-3, 'batch_size': 256, 'gamma': 0.99, 'tau': 0.005},
        {'learning_rate': 1e-4, 'batch_size': 256, 'gamma': 0.98, 'tau': 0.01},
        {'learning_rate': 3e-4, 'batch_size': 128, 'gamma': 0.99, 'tau': 0.02},
        {'learning_rate': 3e-4, 'batch_size': 256, 'gamma': 0.99, 'tau': 0.005},
        {'learning_rate': 3e-4, 'batch_size': 256, 'gamma': 0.95, 'tau': 0.005},
        {'learning_rate': 3e-4, 'batch_size': 256, 'gamma': 0.99, 'tau': 0.01},
        {'learning_rate': 3e-4, 'batch_size': 256, 'gamma': 0.98, 'tau': 0.005},
        {'learning_rate': 3e-4, 'batch_size': 256, 'gamma': 0.99, 'tau': 0.02},
        {'learning_rate': 3e-4, 'batch_size': 512, 'gamma': 0.99, 'tau': 0.005},
        {'learning_rate': 5e-4, 'batch_size': 256, 'gamma': 0.99, 'tau': 0.005}
    ]

    # To store the results
    results = []

    for idx, params in enumerate(parameter_configs):
        print(f"\nRunning configuration {idx+1}/{len(parameter_configs)}: {params}")
        env = gym.make("MountainCarContinuous-v0")
        monitor_dir = f"parameter_search_results/config_{idx}"
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
            'use_sde': True,
            'train_freq': 32,
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
        avg_length = np.mean(lengths)
        results.append({'config_idx': idx, 'params': params, 'avg_length': avg_length})
        print(f"Average Episode Length: {avg_length}")

        env.close()

    # Sort results by average episode length (lower is better)
    results.sort(key=lambda x: x['avg_length'])

    print("\nParameter Search Results:")
    for res in results:
        print(f"Config {res['config_idx']+1}: Avg Length = {res['avg_length']}, Params = {res['params']}")

    # Return the best configuration
    best_config = results[0]['params']
    return best_config

if __name__ == "__main__":
    best_params = run_sac_parameter_search()
