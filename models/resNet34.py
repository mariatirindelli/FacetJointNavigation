from torchvision import models, transforms
from PIL import Image
import torch


class ResNet34:
    """
    Loads the torch ResNet model, modified with two output classes instead of one and provides functions to correctly
    apply transforms to the input images before inference
    """
    def __init__(self, ckpt_path):
        self.transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
        self.ckpt_path = ckpt_path
        self._load_model()

    def transform_image(self, img_array):
        pil_image = Image.fromarray(img_array).convert('RGB')
        tensor_image = self.transform(pil_image).unsqueeze_(0)
        return tensor_image

    def _load_model(self):
        self.model = models.resnet34(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, 2)

        checkpoint = torch.load(self.ckpt_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

