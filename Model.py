from collections import defaultdict
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm 

from MoE import MoE
from losses.MoE_loss import MoELoss, MoELossOutput

class Model(nn.Module):

    def __init__(self, sequence_length, input_dim, hidden_dim, num_classes, input_mean, input_std):
        super(Model, self).__init__()

        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim # used for expert feature extraction
        self.num_classes = num_classes

        self.moe = MoE(sequence_length = self.sequence_length, input_dim=self.input_dim, hidden_dim=self.hidden_dim, num_experts=4, k = 1) 
        self.moe_loss = MoELoss(self.moe)
        self.linear = nn.Linear(input_dim, num_classes)

    def get_load_balance(self):
        return self.moe.get_load_balance()

    def forward(self, x):
        
        moe_output: MoELossOutput = self.moe_loss(x) #(B, T, C)

        x = moe_output.output
        x = torch.mean(x, dim = 1)
        x = self.linear(x) #(B, C)
        
        return x, moe_output

    def reset_histogram(self, device):
        print("Resetting histogram")
        self.moe.reset_histogram(device)

    def save_histogram(self, epoch):
        self.moe.save_histogram(epoch)
        print("Saved Expert Distributions")

