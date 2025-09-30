import numpy as np
from src.ligament_models.base import LigamentFunction
from src.ligament_models.blankevoort import BlankevoortFunction
from src.ligament_models.transformations import sliding_operation

def slide_search(optimal_func: LigamentFunction, test_x: np.ndarray, test_y: np.ndarray, slide_range: tuple[float, float], initial_points: int = 32, depth=4, threshold=1.1):
    optimal_parameters = optimal_func.get_params()
    parameters_names = ['k', 'alpha', 'l_0', 'f_ref']
    param_dict = {parameters_names[i]: optimal_parameters[i] for i in range(len(parameters_names))}

    
    optimal_loss = np.mean((optimal_func(test_x) - test_y)**2)
    threshold_loss = optimal_loss * threshold

    slide_values = np.linspace(slide_range[0], slide_range[1], initial_points) - param_dict['l_0']
    slide_values = np.insert(slide_values, 0, 0)
    slide_values.sort()
    slide_successes = np.zeros(len(slide_values))

    for i,slide_value in enumerate(slide_values):
        slide_parameters = sliding_operation(param_dict, slide_value)
        slid_func = BlankevoortFunction(slide_parameters)
        loss = np.mean((slid_func(test_x) - test_y)**2)

        if loss < threshold_loss:
            slide_successes[i] = 1
        
    max_success_index = np.argmax(slide_successes)
    min_success_index = np.argmin(slide_successes)
    return slide_values[max_success_index], slide_values[min_success_index]

