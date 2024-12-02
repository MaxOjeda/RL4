import os
import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import SAC
from tqdm import tqdm

def busqueda_params(n_configs=10, episodios=1000):
    # Configuracion
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

    resultados = []

    for idx, params in enumerate(parameter_configs):
        print(f"Probando configuracion {idx+1}/{len(parameter_configs)}: {params}")
        env = gym.make("MountainCarContinuous-v0")
        monitor_dir = f"parameter_search_results/config_{idx}"
        os.makedirs(monitor_dir, exist_ok=True)
        monitor_file = os.path.join(monitor_dir, "monitor.csv")
        env = Monitor(env, filename=monitor_file)

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

        model = SAC(**model_params)
        model.learn(total_timesteps=300000, progress_bar=True)

        # monitor.csv
        df = pd.read_csv(monitor_file, comment='#')
        lengths = df['l'].values[:episodios]
        avg_length = np.mean(lengths)
        resultados.append({'config_idx': idx, 'params': params, 'avg_length': avg_length})
        print(f"Largo Episodio Promedio: {avg_length}")

        env.close()

    # Ordenar resultados
    resultados.sort(key=lambda x: x['avg_length'])

    print("\nResultados búsqueda parametros:")
    for res in resultados:
        print(f"Config {res['config_idx']+1}: Avg Length = {res['avg_length']}, Params = {res['params']}")

    # Mejores resultados
    mejor_config = resultados[0]['params']
    return mejor_config

if __name__ == "__main__":
    mejor_params = busqueda_params()
