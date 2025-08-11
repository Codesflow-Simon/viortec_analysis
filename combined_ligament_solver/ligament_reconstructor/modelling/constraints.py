from typing import Dict, List, Union, Callable, Any
import numpy as np

class BetweenConstraint:
    def __init__(self, lower: float, sym: str, upper: float):
        self.lower = lower
        self.sym = sym
        self.upper = upper
    
    def get_dict_function(self) -> Dict[str, Any]:
        return [{'type': 'ineq', 'fun': lambda params: self.to_param(self.sym, params)- self.lower},
                {'type': 'ineq', 'fun': lambda params: self.upper - self.to_param(self.sym, params)}]

    def bounds(self):
        return [self.lower, self.upper]


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

    def get_param_mappings(self, mode: str = 'blankevoort') -> Dict[str, int]:
        """Get parameter name to index mappings for the current mode."""
        if mode == 'trilinear':
            return {
                'k_1': 0,
                'k_2': 1,
                'k_3': 2,
                'l_0': 3,
                'a_1': 4,
                'a_2': 5
            }
        elif mode == 'blankevoort':
            return {
                'alpha': 0,
                'k': 1,
                'l_0': 2,
                'f_ref': 3
            }
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Supported modes: 'trilinear', 'blankevoort'")

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
    
    def get_param_names(self) -> List[str]:
        """Get parameter names for the current mode."""
        if self.mode == 'trilinear':
            return ['k_1', 'k_2', 'k_3', 'l_0', 'a_1', 'a_2']
        elif self.mode == 'blankevoort':
            return ['alpha', 'k', 'l_0', 'f_ref']
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _get_constraints_obj(self) -> List[Dict[str, Any]]:
        if self.mode == 'blankevoort':
            return [
                BetweenConstraint(0.02, 'alpha', 0.12),
                BetweenConstraint(10, 'k', 100),
                BetweenConstraint(40, 'l_0', 50),
                BetweenConstraint(0.0, 'f_ref', 300.0),
            ]
        
        return []

    def get_constraints(self) -> List[Dict[str, Any]]:
        list_list_dicts = [constraint.get_dict_function() for constraint in self._get_constraints_obj()]
        return [item for sublist in list_list_dicts for item in sublist]

    def get_constraints_list(self) -> List[List[Any]]:
        return [constraint.bounds() for constraint in self._get_constraints_obj()]

    


