from sympy import Symbol
from statics_solver.models.statics_model import KneeModel
import yaml
from matplotlib import pyplot as plt



if __name__ == "__main__":
    with open('./statics_solver/mechanics_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    model = KneeModel(config['input_data'], log=False)
    model.build_model()
    model.solve()
    