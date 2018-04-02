import math
import pickle

import numpy as np


class FuzzySystem:
    def __init__(self, params, number_of_rules, dimension):
        params = params.reshape(-1)
        self.alpha = params[-1]
        self.centers = params[:number_of_rules * dimension].reshape(number_of_rules, -1)
        self.widths = params[number_of_rules * dimension:2 * (number_of_rules * dimension)].reshape(number_of_rules, -1)
        self.outs = params[2 * (number_of_rules * dimension): -1].reshape(number_of_rules, -1)
        self.number_of_rules = params.shape[0]
        self.dimension = dimension

    def take_action(self, state):
        state = state.reshape(1, -1)
        temp = np.power(state - self.centers, 2)
        temp = -temp / (2 * np.power(self.widths, 2))
        temp = np.prod(np.exp(temp), axis=1, keepdims=True)
        return math.tanh(self.alpha * (temp.T @ self.outs) / (np.sum(temp)))

    @staticmethod
    def load(params_path):
        with open(params_path, "rb") as f:
            params = pickle.load(f)
        return FuzzySystem(params=params["best"], number_of_rules=params["number_of_rules"],
                           dimension=params["dimension"])
