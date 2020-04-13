
__all__ = ['GenericModel']


class GenericModel(object):
    """
    Generic interface to a DL model
    """
    def __init__(self, ckpt_path):
        """
        :param str ckpt_path: The path to the model .ckpt
        """
        self.ckpt_path = ckpt_path
        self.model = None

    def forward(self, x):
        """
        Runs the model on the input data
        :param x: model input data
        """
        raise NotImplementedError

    def transform_data(self, data):
        """
        :param data: model input data
        """
        raise NotImplementedError

    def _load_model(self):
        raise NotImplementedError
