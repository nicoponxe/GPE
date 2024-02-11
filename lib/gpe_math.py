#  This file contains some math helpers
# ---------------------------------------------------------------------------------------------
import numpy as np

class GPEMath:
    def calculate_angular_velocity(angles, delta_t):
        return np.concatenate(([np.nan], np.diff(angles) / delta_t))

    def calculate_integral(angles, delta_t, velocity):
        integral_values = []
        integral_sum = 0

        for i in range(len(angles) - 1):
            integral_sum += velocity * (angles[i + 1] + angles[i]) * delta_t
            integral_values.append(integral_sum)

        # Add the initial condition (0) to the beginning of the integral_values list
        integral_values.insert(0, 0)

        return integral_values

