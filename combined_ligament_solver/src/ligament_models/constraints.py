from typing import Dict, List, Union, Callable, Any
import numpy as np

class BetweenConstraint:
    def __init__(self, lower: float, param_name: str, upper: float, param_mapping: Dict[str, int]):
        self.lower = lower
        self.param_name = param_name
        self.upper = upper
        self.param_mapping = param_mapping
    
    def get_dict_function(self) -> List[Dict[str, Any]]:
        return [
            {'type': 'ineq', 'fun': lambda params: self.to_param(self.param_name, params) - self.lower},
            {'type': 'ineq', 'fun': lambda params: self.upper - self.to_param(self.param_name, params)}
        ]

    def bounds(self):
        return [self.lower, self.upper]

    def to_param(self, param_name: str, params: np.ndarray) -> float:
        """
        Convert a parameter reference to its actual value.
        
        Args:
            param_name: Parameter name
            params: Parameter array
            
        Returns:
            The parameter value
        """
        if param_name in self.param_mapping:
            return params[self.param_mapping[param_name]]
        else:
            raise ValueError(f"Unknown parameter '{param_name}'. "
                           f"Available parameters: {list(self.param_mapping.keys())}")


class ConstraintManager:
    """
    A class to manage constraints for different ligament models.
    Loads constraint configurations from YAML files.
    """
    
    def __init__(self, constraints_config: Dict[str, Any]):
        """
        Initialize the constraint manager with a configuration.
        
        Args:
            constraints_config: Dictionary containing constraint configuration for a specific mode.
        """
        if constraints_config is None:
            raise ValueError("constraints_config must be provided. Load from YAML in main.py")
            
        self.constraints_config = constraints_config
        
        # Validate that the config has the required structure
        if 'parameters' not in self.constraints_config:
            raise ValueError("constraints_config must contain 'parameters' section")
        if 'constraints' not in self.constraints_config:
            raise ValueError("constraints_config must contain 'constraints' section")
        
        self.param_mapping = self.constraints_config['parameters']
    
    
    def get_param_names(self) -> List[str]:
        """Get parameter names for the current mode."""
        return list(self.param_mapping.keys())

    def _get_constraints_obj(self) -> List[BetweenConstraint]:
        """Create BetweenConstraint objects from the configuration."""
        constraints = []
        constraints_config = self.constraints_config['constraints']
        
        for param_name, constraint_config in constraints_config.items():
            if constraint_config['type'] == 'between':
                constraint = BetweenConstraint(
                    lower=constraint_config['lower'],
                    param_name=param_name,
                    upper=constraint_config['upper'],
                    param_mapping=self.param_mapping
                )
                constraints.append(constraint)
            else:
                raise ValueError(f"Unsupported constraint type: {constraint_config['type']}")
        
        return constraints

    def get_constraints(self) -> List[Dict[str, Any]]:
        """Get constraints in scipy.optimize format."""
        list_list_dicts = [constraint.get_dict_function() for constraint in self._get_constraints_obj()]
        return [item for sublist in list_list_dicts for item in sublist]

    def get_constraints_list(self) -> List[List[Any]]:
        """Get constraints as bounds list."""
        return [constraint.bounds() for constraint in self._get_constraints_obj()]
    
    def get_constraint_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all constraints."""
        descriptions = {}
        constraints_config = self.constraints_config['constraints']
        
        for param_name, constraint_config in constraints_config.items():
            descriptions[param_name] = constraint_config.get('description', 'No description available')
        
        return descriptions