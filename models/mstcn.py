# https://github.com/yabufarha/ms-tcn/blob/master/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from collections import OrderedDict
from models import GenericModel

__all__ = ['MSTCN']

REGRESSION = 1
MULTI_CLASS = 2
BINARY = 3


class MSTCN(GenericModel):
    """
    Loads the torch ResNet model, modified with two output classes instead of one and provides functions to correctly
    apply transforms to the input images before inference
    """
    def __init__(self, ckpt_path):

        super(MSTCN, self).__init__(ckpt_path)

        self.ckpt_path = ckpt_path
        self.num_fmaps = 32
        self.num_layers = 9
        self.num_stages = 3
        self.input_features = 1
        self.num_classes = 9
        self.problem = MULTI_CLASS
        self._load_model()

    def forward(self, x):
        return self.model.forward(x)

    def transform_data(self, data):
        return data

    def _load_model(self):

        self.model = MultiStageModel(num_fmaps=self.num_fmaps, num_layers=self.num_layers, num_stages=self.num_stages,
                                     input_features=self.input_features, num_classes=self.num_classes)
        checkpoint = torch.load(self.ckpt_path)
        state_dict = self._adjust_dict(checkpoint['state_dict'])

        # load params
        self.model.load_state_dict(state_dict)

    def _set_params(self, params):
        """
        Retrieve the model parameters from the checkpoints. If the parameters are not saved in the checkpoints, the
        value should be entered manually - this function is specific to the way you saved the parameters in the
        checkpoints.
        """
        self.num_fmaps = params["msctn_f_maps"]
        self.num_layers = params["msctn_layers"]
        self.num_stages = params["msctn_stages"]
        self.num_classes = params["out_features"]

        if params["problem"] == "multiclass":
            self.problem = MULTI_CLASS
        elif params["problem"] == "regression":
            self.problem = REGRESSION
            self.num_classes = 1

        if params["modality"] == "imgProb_force":
            self.input_features = 2
        else:
            self.input_features = 1

    @staticmethod
    def _adjust_dict(state_dict):
        """
        remove `model.` prefix from the saved state_dict. This happens when using pytorch lighting for the training
        It is not needed if no prefix is saved in the model dict.
        """
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[6:]  # remove `model.`
            new_state_dict[name] = v
        return new_state_dict


class MultiStageModel(nn.Module):
    """
    Multistage 1D convolutional network

    :ivar int num_f_maps: Number of channels in each convolution layer
    :ivar int num_layers: Number of convolution layers in each stage
    :ivar int num_stages: Number of stages in the architecture
    :ivar int input_features: Number of input features in the input data (size of the input features vector)
    :ivar int num_classes: number of classes (1 if it is a regression problem)
    :ivar problem: problem type. It can be either a multi-class classification (MULTI_CLASS), a binary
    classification problem (BINARY) or a regression problem (REGRESSION)
    :ivar bool causal: whether the convolution is causal or not (a-causal)
    """
    def __init__(self, num_fmaps, num_layers, num_stages, input_features,  num_classes=1, problem=MULTI_CLASS,
                 causal=False):

        super(MultiStageModel, self).__init__()

        self.num_fmaps = num_fmaps  # 64
        self.num_layers = num_layers  # 10
        self.num_stages = num_stages  # 4
        self.input_features = input_features   # 2048
        self.num_classes = num_classes  # 7
        self.problem = problem
        if self.problem == REGRESSION or self.problem == BINARY:
            self.num_classes = 1

        self.causal = causal

        self.stage1 = SingleStageModel(num_fmaps=self.num_f_maps,
                                       num_layers=self.num_layers,
                                       input_features=self.input_features,
                                       num_classes=self.num_classes,
                                       causal=self.causal)

        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_fmaps=self.num_fmaps,
                                                                    num_layers=self.num_layers,
                                                                    input_features=self.num_classes,
                                                                    num_classes=self.num_classes,
                                                                    causal=self.causal))
                                     for s in range(self.num_stages-1)])

        print("MTCN model initialized: num stages: {} -- num layers: {} -- num features maps: {} -- input features: {}"
              "-- problem: {} -- num classes: {}".format(self.num_stages, self.num_layers, self.num_fmaps,
                                                         self.input_features, self.problem, self.num_classes))

    def forward(self, x):

        out = self.stage1(x)
        outputs = out.unsqueeze(0)

        for s in self.stages:
            if self.problem == REGRESSION:
                out = s(torch.sigmoid(out))
            else:
                out = s(F.softmax(out, dim=1))
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SingleStageModel(nn.Module):
    """
        Single stage 1D convolutional network

        :ivar int num_f_maps: Number of channels in each convolution layer
        :ivar int num_layers: Number of convolution layers in each stage
        :ivar int input_features: Number of input features in the input data (size of the input features vector)
        :ivar int num_classes: number of classes (1 if it is a regression problem)
        :ivar bool causal: whether the convolution is causal or not (a-causal)
        """
    def __init__(self, num_fmaps, num_layers, input_features, num_classes, causal=False):
        super(SingleStageModel, self).__init__()

        # 1x1 convolution to remap the input data to a N x num_fmaps array, where N is the sequence length. The kernel
        # size is 1, therefore it is already causal
        self.conv_1x1 = nn.Conv1d(input_features, num_fmaps, 1)  # This is already causal cause kernel size is 1

        # 1D dilated convolution layers with kernel size 3
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(dilation=2 ** i,
                                                                        in_channels=num_fmaps,
                                                                        out_channels=num_fmaps,
                                                                        kernel_size=3,
                                                                        causal=causal))
                                     for i in range(num_layers)])

        # 1x1 convolution to remap the output features to a N x num_classes array, where N is the sequence length.
        # The kernel size is 1, therefore it is already causal
        self.conv_out = nn.Conv1d(num_fmaps, num_classes, 1)  # This is already causal cause kernel size is 1

    def forward(self, x):
        """
        :param x: The input tensor. It's size should be B x M x N where B is the batch size M is the number of input
        features and N is the sequence length. The sequence length can be any number while the number of features M must
        coincide with the one defined at construction time (i.e. dim)
        :return: the processed input
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
        return x + out
