from .base import LigamentFunction
import numpy as np
from sympy import symbols, Piecewise

class BlankevoortFunction(LigamentFunction):
    def __init__(self, params: np.ndarray):
        """
        Initialize with parameters [alpha, k, l_0, f_ref]
        """
        if len(params) != 4:
            raise ValueError(f"BlankevoortFunction requires 4 parameters: [alpha, k, l_0, f_ref], got {params}")
        super().__init__(params)

    def sympy_implementation(self):
        """
        Returns symbolic expression for Blankevoort function using sympy.
        """
        x, alpha, k, l_0, f_ref = symbols('x alpha k l_0 f_ref')

        transition_length = l_0 * (1 + alpha)
        transition_elongation = transition_length - l_0

        force_expr = lambda x_: Piecewise(
            (0, x_ < l_0),
            (k * (x_ - l_0)**2 / (2 * transition_elongation), x_ <= transition_length),
            (k * (transition_elongation)**2 / (2 * transition_elongation) + k * (x_ - transition_length), True)
        )

        relative_force = force_expr(x)
        total_force = relative_force - f_ref
        return total_force

    def get_param_symbols(self):
        return symbols('alpha k l_0 f_ref')


## Keep this around for legacy


class TrilinearFunction(LigamentFunction):
    def __init__(self, params: np.ndarray):
        """
        Initialize with parameters [k_1, k_2, k_3, l_0, a_1, a_2]
        """
        if len(params) != 6:
            raise ValueError("TrilinearFunction requires 6 parameters: [k_1, k_2, k_3, l_0, a_1, a_2]")
        super().__init__(params)

    def sympy_implementation(self):
        """
        Returns symbolic expression for trilinear function using sympy.
        """
        x, k_1, k_2, k_3, l_0, a_1, a_2 = symbols('x k_1 k_2 k_3 l_0 a_1 a_2')
        x_1 = a_1 * l_0
        x_2 = a_2 * l_0

        # Define piecewise function
        expr = Piecewise(
            (0, x < l_0),
            (k_1 * (x - l_0), x < x_1),
            (k_1 * (x_1 - l_0) + k_2 * (x - x_1), x < x_2),
            (k_1 * (x_1 - l_0) + k_2 * (x_2 - x_1) + k_3 * (x - x_2), True)
        )

        return expr

    def get_param_symbols(self):
        return symbols('k_1 k_2 k_3 l_0 a_1 a_2')


