import torch

checkpoint = torch.load("/media/maria/Elements/Maria/Submissions/IROS2020/Experiments/models/TCN_imgProb.ckpt")
state_dict = checkpoint['state_dict']

input("press")