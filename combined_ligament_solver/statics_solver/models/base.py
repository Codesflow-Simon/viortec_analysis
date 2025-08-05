class AbstractModel:
    def __init__(self, data, log=True):
        pass

    def build_model(self):
        raise NotImplementedError("Subclasses must implement this method")

    def solve(self):
        raise NotImplementedError("Subclasses must implement this method")
    
    def plot_model(self):
        raise NotImplementedError("Subclasses must implement this method")
    