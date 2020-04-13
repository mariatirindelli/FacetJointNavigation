from models import GenericModel
import os
from torchvision import models, transforms
from PIL import Image
import torch

__all__ = ['ResNet18', 'ResNet34']


class ResNet18(GenericModel):
    """
    Loads the torch ResNet18 model, modified with two output classes instead of one and provides functions to correctly
    apply transforms to the input images before inference
    """
    def __init__(self, ckpt_path):
        super(ResNet18, self).__init__(ckpt_path)
        self.transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
        self._load_model()

    def transform_data(self, data):
        """
        Transform the input image into a 1 x 224 x 224 x 3 tensor, compatible with the network input
        :param ndarray data: input image of arbitrary width and height. It can be either and RGB or a gray image.
        """
        pil_image = Image.fromarray(data).convert('RGB')
        tensor_image = self.transform(pil_image).unsqueeze_(0)
        return tensor_image

    def forward(self, x):
        """
        :param torch.tensor x: input tensor of size BS x 224 x 224 x 3 where BS is the batch size.
        :return: The network output given the input data
        """
        return self.model(x)

    def _load_model(self):

        if not os.path.exists(self.ckpt_path):
            raise FileExistsError("Ckpt Path: {} not exists".format(self.ckpt_path))

        self.model = models.resnet18(pretrained=False)
        num_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_features, 2)

        checkpoint = torch.load(self.ckpt_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])


class ResNet34(GenericModel):
    """
       Loads the torch ResNet34 model, modified with two output classes instead of one and provides functions to correctly
       apply transforms to the input images before inference
       """

    def __init__(self, ckpt_path):
        super(GenericModel, self).__init__(ckpt_path)
        self.transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
        self._load_model()

    def transform_data(self, data):
        """
        Transform the input image into a 1 x 224 x 224 x 3 tensor, compatible with the network input
        :param ndarray data: input image of arbitrary width and height. It can be either and RGB or a gray image.
        """
        pil_image = Image.fromarray(data).convert('RGB')
        tensor_image = self.transform(pil_image).unsqueeze_(0)
        return tensor_image

    def forward(self, x):
        """
        :param torch.tensor x: input tensor of size BS x 224 x 224 x 3 where BS is the batch size.
        :return: The network output given the input data
        """
        return self.model(x)

    def _load_model(self):
        if not os.path.exists(self.ckpt_path):
            raise FileExistsError("Ckpt Path: {} not exists".format(self.ckpt_path))

        self.model = models.resnet34(pretrained=False)
        num_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_features, 2)

        checkpoint = torch.load(self.ckpt_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
