from models import *
import torch
from torch import nn
from PIL import Image
from torchvision import transforms
import numpy as np


class MyLayer(nn.Module):
    def __init__(self):
        super(MyLayer, self).__init__()
        self.transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])

    def forward(self, x):
        tensor_image = x.unsqueeze_(0).float()
        return tensor_image


gray_image = Image.open("/home/maria/Desktop/data/Maria_T 2.png")
pil_image = gray_image.convert('RGB')
pil_image.save("/home/maria/Desktop/data/image.jpg")

dummy_input = torch.rand(1, 3, 224, 224)
cnn_model_ckpt = "/home/maria/Desktop/data/vertebraModel/ResNet18.pt"
cnn_model = ResNet18(cnn_model_ckpt)

# model = nn.Sequential(
#     MyLayer(),
#     cnn_model.model
# )

traced_model = torch.jit.trace(cnn_model.model, dummy_input)
traced_model.save("/home/maria/Desktop/data/vertebraModel/traced.pt")

model = torch.jit.load("/home/maria/Desktop/data/vertebraModel/traced.pt")
input_image = np.array(gray_image)[0:224, 0:224]
#stacked_img = np.stack((input_image,)*3, axis=0)
stacked_img1 = np.expand_dims(np.stack((input_image,)*3, axis=0), axis=0)
stacked_img2 = np.expand_dims(np.stack((input_image,)*3, axis=0), axis=0)
stacked_img = np.concatenate([stacked_img1, stacked_img2], axis=0)

in_tensor = torch.from_numpy(stacked_img).float()
a = model.forward(in_tensor)
print()
