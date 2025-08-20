class State:
    def __init__(self):
        self.dict = None

    def read_config(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return self.config