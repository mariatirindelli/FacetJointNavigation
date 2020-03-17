

class GenericStep(object):
    def __init__(self):
        self.name = "GENERIC"
        self.device = "cpu"

    def gpu_acceleration(self):
        self.device = "cuda"

    def process(self, server_handler):
        raise NotImplementedError

    def release_models(self):
        raise NotImplementedError

    def _load_models_to_device(self, device):
        raise NotImplementedError