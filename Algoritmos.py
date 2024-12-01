import random
import numpy as np

from FeatureExtractor import FeatureExtractor

class Sarsa:

    def __init__(self, num_actions: int, epsilon: float, alpha: float, gamma: float):
        self.__num_actions = num_actions
        self.__epsilon = epsilon
        self.__alpha = alpha
        self.__gamma = gamma
        self.__feature_extractor = FeatureExtractor()
        self.__num_features = self.__feature_extractor.num_of_features
        self.__weights = np.zeros(self.__num_features)

    def sample_action(self, observation):
        if random.random() < self.__epsilon:
            return random.randrange(self.__num_actions)
        return self.argmax(observation)

    def argmax(self, observation):
        max_value = float('-inf')
        best_actions = []
        for action in range(self.__num_actions):
            q_value = self.__get_q_estimate(observation, action)
            if q_value > max_value:
                max_value = q_value
                best_actions = [action]
            elif q_value == max_value:
                best_actions.append(action)
        return random.choice(best_actions)

    def __get_q_estimate(self, observation, action):
        x = self.__feature_extractor.get_features(observation, action)
        return np.dot(self.__weights, x)

    def learn(self, observation, action, reward, next_observation, next_action, done):
        x = self.__feature_extractor.get_features(observation, action)
        x_next = self.__feature_extractor.get_features(next_observation, next_action)
        q_current = np.dot(self.__weights, x)
        q_next = np.dot(self.__weights, x_next) if not done else 0.0
        delta = reward + self.__gamma * q_next - q_current
        self.__weights += self.__alpha * delta * x


class QLearning:

    def __init__(self, num_actions: int, epsilon: float, alpha: float, gamma: float):
        self.__num_actions = num_actions
        self.__epsilon = epsilon
        self.__alpha = alpha
        self.__gamma = gamma
        self.__feature_extractor = FeatureExtractor()
        self.__num_features = self.__feature_extractor.num_of_features
        self.__weights = np.zeros(self.__num_features)

    def sample_action(self, observation):
        if random.random() < self.__epsilon:
            return random.randrange(self.__num_actions)
        return self.argmax(observation)

    def argmax(self, observation):
        max_value = float('-inf')
        best_actions = []
        for action in range(self.__num_actions):
            q_value = self.__get_q_estimate(observation, action)
            if q_value > max_value:
                max_value = q_value
                best_actions = [action]
            elif q_value == max_value:
                best_actions.append(action)
        return random.choice(best_actions)

    def __get_q_estimate(self, observation, action):
        x = self.__feature_extractor.get_features(observation, action)
        return np.dot(self.__weights, x)

    def learn(self, observation, action, reward, next_observation, done):
        x = self.__feature_extractor.get_features(observation, action)
        if not done:
            q_next = max(
                self.__get_q_estimate(next_observation, a)
                for a in range(self.__num_actions)
            )
        else:
            q_next = 0.0
        q_current = np.dot(self.__weights, x)
        delta = reward + self.__gamma * q_next - q_current
        self.__weights += self.__alpha * delta * x


class SarsaLambda:

    def __init__(self, num_actions: int, epsilon: float, alpha: float, gamma: float, lambd: float):
        self.__num_actions = num_actions
        self.__epsilon = epsilon
        self.__alpha = alpha
        self.__gamma = gamma
        self.__lambda = lambd
        self.__feature_extractor = FeatureExtractor()
        self.__num_features = self.__feature_extractor.num_of_features
        self.__weights = np.zeros(self.__num_features)
        self.__e_trace = np.zeros(self.__num_features)

    def sample_action(self, observation):
        if random.random() < self.__epsilon:
            return random.randrange(self.__num_actions)
        return self.argmax(observation)

    def argmax(self, observation):
        max_value = float('-inf')
        best_actions = []
        for action in range(self.__num_actions):
            q_value = self.__get_q_estimate(observation, action)
            if q_value > max_value:
                max_value = q_value
                best_actions = [action]
            elif q_value == max_value:
                best_actions.append(action)
        return random.choice(best_actions)

    def __get_q_estimate(self, observation, action):
        x = self.__feature_extractor.get_features(observation, action)
        return np.dot(self.__weights, x)

    def learn(self, observation, action, reward, next_observation, next_action, done):
        x = self.__feature_extractor.get_features(observation, action)
        x_next = self.__feature_extractor.get_features(next_observation, next_action)
        q_current = np.dot(self.__weights, x)
        q_next = np.dot(self.__weights, x_next) if not done else 0.0
        delta = reward + self.__gamma * q_next - q_current

        # Update eligibility traces
        self.__e_trace = self.__gamma * self.__lambda * self.__e_trace + x

        # Update weights
        self.__weights += self.__alpha * delta * self.__e_trace

        if done:
            # Reset eligibility traces at the end of each episode
            self.__e_trace = np.zeros(self.__num_features)
