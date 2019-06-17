import numpy as np
from pyitlib import discrete_random_variable as drv


def info_gain(feature_vals: np.ndarray, y_vals: np.ndarray) -> float:
    h_y = drv.entropy(y_vals)
    h_y_given_x = drv.entropy_conditional(y_vals, feature_vals)
    return h_y - h_y_given_x