from typing import Dict, List, Union, Callable, Any
import numpy as np

class ConstraintManager:
    """
    A class to manage constraints for different ligament models.
    Handles parameter mapping and constraint generation elegantly.
    """
    
    def __init__(self, mode: str = 'blankevoort'):
        """
        Initialize the constraint manager with a specific mode.
        
        Args:
            mode: The ligament model mode ('trilinear' or 'blankevoort')
        """
        self.mode = mode
    
    def get_param_mappings(self) -> Dict[str, int]:
        """Get parameter name to index mappings for the current mode."""
        if self.mode == 'trilinear':
            return {
                'k_1': 0,
                'k_2': 1,
                'k_3': 2,
                'l_0': 3,
                'a_1': 4,
                'a_2': 5
            }
        elif self.mode == 'blankevoort':
            return {
                'alpha': 0,
                'k': 1,
                'l_0': 2,
                'l_ref': 3
            }
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Supported modes: 'trilinear', 'blankevoort'")
    
    def get_param_names(self) -> List[str]:
        """Get parameter names for the current mode."""
        if self.mode == 'trilinear':
            return ['k_1', 'k_2', 'k_3', 'l_0', 'a_1', 'a_2']
        elif self.mode == 'blankevoort':
            return ['alpha', 'k', 'l_0', 'l_ref']
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def to_param(self, a: Union[str, float, int], params: np.ndarray) -> float:
        """
        Convert a parameter reference to its actual value.
        
        Args:
            a: Parameter reference (string name or numeric value)
            params: Parameter array
            
        Returns:
            The parameter value
        """
        if isinstance(a, str):
            if a in self.get_param_mappings():
                return params[self.get_param_mappings()[a]]
            else:
                raise ValueError(f"Unknown parameter '{a}' for mode '{self.mode}'. "
                               f"Available parameters: {list(self.get_param_mappings().keys())}")
        else:
            return float(a)
    
    def greater_than(self, a: Union[str, float, int], b: Union[str, float, int]) -> Dict[str, Any]:
        """
        Create a greater than constraint: a > b
        
        Args:
            a: Left side of inequality (parameter name or value)
            b: Right side of inequality (parameter name or value)
            
        Returns:
            Constraint dictionary for scipy.optimize
        """
        def constraint_func(params):
            return self.to_param(a, params) - self.to_param(b, params)
        
        return {'type': 'ineq', 'fun': constraint_func}
    
    def n_a_greater_m_b(self, n: float, a: Union[str, float, int], 
                       m: float, b: Union[str, float, int]) -> Dict[str, Any]:
        """
        Create a constraint: n*a > m*b
        
        Args:
            n: Multiplier for parameter a
            a: First parameter (parameter name or value)
            m: Multiplier for parameter b
            b: Second parameter (parameter name or value)
            
        Returns:
            Constraint dictionary for scipy.optimize
        """
        def constraint_func(params):
            return n * self.to_param(a, params) - m * self.to_param(b, params)
        
        return {'type': 'ineq', 'fun': constraint_func}

    def get_constraints(self) -> List[Dict[str, Any]]:

        """Get the default constraints for the current mode."""




        if self.mode == 'trilinear':
            return [
                # k_3 bounds
                self.greater_than('k_3', 500),
                self.greater_than(5000, 'k_3'),  # Added upper bound for stability
                
                # k_2 is generally twice k_1 (with some flexibility)
                self.n_a_greater_m_b(2, 'k_1', 1, 'k_2'),
                self.n_a_greater_m_b(1, 'k_2', 2, 'k_1'),
                
                # l_0 must be positive
                self.greater_than('l_0', 0),
                
                # a_1 bounds to prevent optimization explosion
                self.greater_than('a_1', 1.03),
                
                # a_2 must be greater than a_1
                self.greater_than('a_2', 'a_1'),
                
                # Additional stability constraints
                self.greater_than('k_1', 100),
                self.greater_than('k_2', 100),
            ]
        
        elif self.mode == 'blankevoort':
            return [
                # alpha must be positive
                self.greater_than('alpha', 0.02),
                self.greater_than(0.12, 'alpha'),
                
            #     # k must be positive
                self.greater_than('k', 10),
                self.greater_than(100, 'k'),
                
            #     # l_0 must be positive
                self.greater_than('l_0', 30),
                self.greater_than(50, 'l_0'),

                self.greater_than('l_ref', 0.0),
                self.greater_than(500.0, 'l_ref'),
            ]
        
        return []

