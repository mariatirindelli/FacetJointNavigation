# https://github.com/yabufarha/ms-tcn/blob/master/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class MSTCN:
    """
    Loads the torch ResNet model, modified with two output classes instead of one and provides functions to correctly
    apply transforms to the input images before inference
    """
    def __init__(self, ckpt_path, num_layers=9, num_stages=3, num_fmaps=32, dim=1, num_classes=9):
        self.ckpt_path = ckpt_path
        self._load_model()

    def _load_model(self):
        device_ids = list(range(0, 1))
        self.model = MultiStageModel(num_layers=9, num_stages=3, num_fmaps=32, dim=1, num_classes=9)
        checkpoint = torch.load(self.ckpt_path)
        state_dict = checkpoint['state_dict']

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[6:]  # remove `model.`
            new_state_dict[name] = v
        # load params
        self.model.load_state_dict(new_state_dict)

class MultiStageModel(nn.Module):
    """
    :ivar num_stages Number of stages in the architecture
    :ivar num_layers Number of convolution layers in each stage
    :ivar num_f_maps Number of channels in each convolution layer
    :ivar dim Number of features in the input data (size of the input features vector)
    :ivar num_classes: number of classes (1 if it is a regression problem)

        """
    def __init__(self, num_layers, num_stages, num_fmaps, dim,  num_classes):

        self.num_stages = num_stages  # 4
        self.num_layers = num_layers  # 10
        self.num_f_maps = num_fmaps  # 64
        self.dim = dim   # 2048
        self.num_classes = num_classes  # 7
        self.problem = "multiclass"
        self.causal = False

        if self.problem == "regression":
            self.num_classes = 1

        print(f"num_stages: {self.num_stages}, num_layers: {self.num_layers}, num_f_maps: {self.num_f_maps}, dim: {self.dim}")
        super(MultiStageModel, self).__init__()

        self.stage1 = SingleStageModel(self.num_layers, self.num_f_maps, self.dim, self.num_classes, self.causal)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(self.num_layers, self.num_f_maps, self.num_classes,
                                                                    self.num_classes, self.causal))
                                     for s in range(self.num_stages-1)])

    # TODO: mask can be removed
    def forward(self, x):
        out = self.stage1(x)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            if self.problem == "regression":
                out = s(torch.sigmoid(out))
            else:
                out = s(F.softmax(out, dim=1))
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes, causal=False):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)  # This is already causal cause kernel size is 1
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps, kernel_size=3,
                                                                        causal=causal)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)  # This is already causal cause kernel size is 1

    def forward(self, x):
        """
        :param x: The input tensor. It's size should be B x M x N where B is the batch size M is the number of input
        features and N is the sequence length. The sequence length can be any number while the number of features M must
        coincide with the one defined at construction time (i.e. dim)
        :param mask: TODO: check what this mask does - it can be probably removed
        :return:
        """
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, kernel_size=3, causal=False):
        super(DilatedResidualLayer, self).__init__()

        self.causal = causal
        self.kernel_size = kernel_size
        if causal:
            padding = (kernel_size - 1) * dilation
        else:
            padding = int(kernel_size/2) * dilation
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)  # This is already causal because kernel size is 1
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        if self.causal:
            out = out[:, :, :-self.conv_dilated.padding[0]]

        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out)
