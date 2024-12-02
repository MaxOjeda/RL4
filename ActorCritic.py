import numpy as np
import random
from FeatureExtractor import FeatureExtractorSAC

class ActorCritic:
    def __init__(self, alpha_v, alpha_pi, gamma):
        self.alpha_v = alpha_v
        self.alpha_pi = alpha_pi
        self.gamma = gamma

        self.feature_extractor = FeatureExtractorSAC()
        self.num_features = self.feature_extractor.num_of_features

        # Inicializar pesos
        self.w_mu = np.zeros(self.num_features)
        self.w_sigma = np.zeros(self.num_features)
        #self.w_sigma = np.ones(self.num_features) * (-1)
        self.theta = np.zeros(self.num_features)

    def select_action(self, observation):
        phi_s = self.feature_extractor.get_features(observation)
        log_sigma = np.dot(self.w_sigma, phi_s)

        # Clipear valores por erroes de None
        max_log_sigma = 2
        min_log_sigma = -20
        log_sigma = np.clip(log_sigma, min_log_sigma, max_log_sigma)
        sigma = np.exp(log_sigma)

        sigma = np.clip(sigma, 1e-4, 10.0)

        mu = np.dot(self.w_mu, phi_s)

        action = random.gauss(mu, sigma)

        # Clip
        action = np.clip(action, -1.0, 1.0)
        return action, mu, sigma, phi_s

    def learn(self, observation, action, reward, next_observation, done, mu, sigma, phi_s):
        # Computar caracteristicas next
        phi_s_next = self.feature_extractor.get_features(next_observation)

        V_s = np.dot(self.theta, phi_s)
        V_s_next = np.dot(self.theta, phi_s_next) if not done else 0.0

        # Error
        delta = reward + self.gamma * V_s_next - V_s

        # Actualizar con weight decay
        weight_decay = 1e-4
        self.theta += self.alpha_v * (delta * phi_s - weight_decay * self.theta)

        sigma = np.clip(sigma, 1e-4, 10.0) 
        sigma_squared = sigma ** 2

        # Para evitar division por 0
        if sigma_squared == 0:
            sigma_squared = 1e-8

        grad_logp_mu = ((action - mu) / sigma_squared) * phi_s
        grad_logp_sigma = (((action - mu) ** 2) / sigma_squared - 1) * phi_s

        max_grad = 5.0
        grad_logp_mu = np.clip(grad_logp_mu, -max_grad, max_grad)
        grad_logp_sigma = np.clip(grad_logp_sigma, -max_grad, max_grad)

        # Actualizar pesos
        self.w_mu += self.alpha_pi * (delta * grad_logp_mu - weight_decay * self.w_mu)
        self.w_sigma += self.alpha_pi * (delta * grad_logp_sigma - weight_decay * self.w_sigma)