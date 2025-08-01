


class StaticsModel:
    def __init__(self, config_dict):
        self.config_dict = config_dict
        self.frames = {}
        self.bodies = {}
        self.joints = {}
        self.forces = {}
        self.constraints = {}
        self.unknowns = {}
        self.equations = {}
        self.solutions = {}
        self.results = {}
        