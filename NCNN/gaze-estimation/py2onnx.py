import torch
from datasets.unity_eyes import UnityEyesDataset
from models.eyenet import EyeNet
import os
import numpy as np
import cv2
from util.preprocess import gaussian_2d
from matplotlib import pyplot as plt
from util.gaze import draw_gaze



import os
import argparse
from torch.autograd import Variable
import torch
import os

import torch
from datasets.unity_eyes import UnityEyesDataset
from torch.utils.data import DataLoader
from models.eyenet import EyeNet
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
import cv2
import argparse

# Set up pytorch
torch.backends.cudnn.enabled = False
torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device', device)



device = torch.device('cpu')
dataset = UnityEyesDataset()
checkpoint = torch.load('checkpoint.pt', map_location=device)
nstack = checkpoint['nstack']
nfeatures = checkpoint['nfeatures']
nlandmarks = checkpoint['nlandmarks']
eyenet = EyeNet(nstack=nstack, nfeatures=nfeatures, nlandmarks=nlandmarks).to(device)
eyenet.load_state_dict(checkpoint['model_state_dict'])


print("=====> load pytorch checkpoint...")


print("=====> convert pytorch model to onnx...")
dummy_input = Variable(torch.randn(3, 160, 96))
# dummy_input = torch.randn(3, 160, 96)

input_names = ["input"]
output_names = ["output"]
# torch.onnx.export(eyenet, dummy_input, "eyenet.onnx",opset_version=11)

# torch.onnx.export(eyenet, dummy_input, "{}_{}_{}.onnx".format(MODEL_TYPE, INPUT_SIZE, WIDTH_FACTOR), \
#     verbose=False, input_names=input_names, output_names=output_names)



torch.onnx.export(eyenet, dummy_input,"eyenet-1.onnx",\
     verbose=False, input_names=input_names, output_names=output_names)    