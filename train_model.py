import argparse

import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

from collections import defaultdict
import csv

import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm 
import time

from losses.MoE_loss import MoELoss, MoELossOutput

from skimage.util import view_as_windows

from Model import Model

class MoELossScheduler:
    def __init__(self,
                 total_epochs: int,
                 lb_max: float = 0.05,
                 lb_min: float = 0.0,
                 ent_max: float = 0.01,
                 ent_min: float = 0.0,
                 mode: str = "linear"):
        self.total_epochs = total_epochs
        self.lb_max = lb_max
        self.lb_min = lb_min
        self.ent_max = ent_max
        self.ent_min = ent_min
        self.mode = mode

    def _linear(self, start, end, epoch):
        ratio = min(epoch / self.total_epochs, 1.0)
        return start + ratio * (end - start)

    def get_weights(self, epoch: int) -> tuple[float, float]:
        if self.mode == "linear":
            lb_weight = self._linear(self.lb_max, self.lb_min, epoch)
            ent_weight = self._linear(self.ent_min, self.ent_max, epoch)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
        return lb_weight, ent_weight



def accuracy(model, loader, device, epoch, draw_first = False):

    model.reset_histogram(device)

    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        first = True if draw_first else False
        for data in loader:
            images, labels = data
            images = sliding_window(images)
            images, labels = images.to(device), labels.to(device)
            outputs, moe_outputs = model(images)
            if first:
                expert_positions = moe_outputs.expert_positions
                draw_specializations(images, expert_positions, epoch)
                # draw_heatmaps(labels, expert_positions, epoch)
                first = False
                print(moe_outputs.load_balance)
            probs = nn.functional.softmax(outputs, dim = -1)
            pred_labels = torch.argmax(probs, dim = -1)
            correct += (pred_labels == labels).sum().item()
            total += torch.numel(labels)
    model.save_histogram(epoch)

    accuracy = (correct / total) * 100
    return accuracy


def sliding_window(inputs, window_size = 5, pad = 1):

    inputs = inputs.squeeze(1)
    B, T, C = inputs.shape

    inputs = np.array(inputs)
    inputs = np.pad(inputs, ((0, 0), (pad, pad), (pad, pad)), mode='reflect')

    inputs = view_as_windows(inputs, (1, window_size, window_size), (1, window_size, window_size))
    inputs = inputs.reshape(inputs.shape[0], inputs.shape[1] * inputs.shape[2], -1)
    return torch.Tensor(inputs)

    # tokens = []
    
    # for i in range(T - window_size + 1):
    #     window = inputs[:, i:i + window_size, :] # B x window_size x C
    #     window = window.reshape(B, window_size * C) 
    #     tokens.append(window)

    # inputs = torch.stack(tokens, dim = 1)
    # return inputs

# explainability
def draw_specializations(images, expert_positions, epoch):
    images = images.view(-1, images.shape[2])
    print(images.shape)

    num_experts = 4

    
    fig, ax = plt.subplots(num_experts)

    for expert in range(num_experts):
        token_positions = torch.cat(expert_positions[expert], dim = 0)
        if(token_positions.allclose(torch.zeros_like(token_positions))):
            tokens = torch.zeros(5, 5)
        else:    
            # tokens = torch.cat(tokens, dim = 0)
            tokens = images[token_positions]
            tokens = tokens.detach().cpu()
            # tokens = tokens * self.input_std + self.input_mean
            tokens = torch.mean(tokens, dim = 0)
            tokens = tokens.reshape(5, 5)

        ax[expert].imshow(tokens, cmap='gray', vmin = 0, vmax = 1)
        ax[expert].set_title(f"Expert {expert}'s Tokens")

    fig.tight_layout()
    plt.savefig(f"specializations/expert_tokens_epoch_{epoch}.png")
    plt.close()
    print("Drew Tokens!")   

def draw_heatmaps(labels, expert_positions, epoch):
    num_experts = len(expert_positions.keys())

    heatmap = torch.zeros(10, num_experts)
    for expert in range(num_experts):
        token_positions = torch.cat(expert_positions[expert], dim = 0)

        expert_labels = labels[token_positions].detach().cpu()
        print(expert_labels)
        bincount = torch.bincount(expert_labels, minlength = 10) # how many labels for this expert
        heatmap[:, expert] += bincount

    heatmap = heatmap / heatmap.sum(dim = 1) 
    pos = ax.imshow(heatmap, cmap='viridis')

    ax.set_title("Expert Usage Heatmap by Class")
    fig.colorbar(pos, ax = ax)
    plt.savefig(f"class_heatmaps/heatmap_epoch_{epoch}.png")
    plt.close()




def train(model, train_loader, test_loader, criterion, optimizer, epoch, device, writer, training_stats_filename, scheduler):

    model.train()
    running_loss = 0

    torch.cuda.reset_max_memory_allocated()
    start = time.time()
    
    num_batches = len(train_loader)

    # training_stats
    total_ce_loss = 0
    total_lb_loss = 0
    total_routing_entropy = 0

    lb_weight, ent_weight = scheduler.get_weights(epoch)

    with tqdm.tqdm(train_loader, total=len(train_loader)) as tepoch:
        for i, data in enumerate(tepoch):
            inputs, labels = data

            inputs = sliding_window(inputs)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, moe_outputs = model(inputs)
            ce_loss = criterion(outputs, labels)
            load_balance = moe_outputs.load_balance
            lb_loss = moe_outputs.lb_loss
            routing_entropy =  moe_outputs.routing_entropy 

            final_loss = ce_loss + ent_weight * routing_entropy + lb_weight * lb_loss

            tepoch.set_description(f"Epoch {epoch}")
            tepoch.set_postfix(loss=final_loss.item())

            final_loss.backward()
            optimizer.step()

            # update training stats
            total_ce_loss += ce_loss
            total_lb_loss += lb_loss
            total_routing_entropy += routing_entropy

            # print statistics
            running_loss += final_loss.item()
            if i % 200==199:
                print(f"Cross-Entropy Loss: {ce_loss}")
                print(f"Load Balance Loss: {lb_loss}")
                # print(f"Balances: {load_balance}")
                print(f"Routing Entropy: {routing_entropy}")
                print("[epoch %d, iter %5d] loss: %.3f"%(epoch,i+1,running_loss/200))
                running_loss=0.0
    


    train_acc = accuracy(model, train_loader, device, epoch, draw_first = False)
    test_acc = accuracy(model, test_loader, device, epoch, draw_first = True)
    print("Epoch %d: Train Accuracy: %.3f, Test Accuracy: %.3f"%(epoch+1,train_acc,test_acc))


    end = time.time()
    max_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)

    with open(training_stats_filename, "a") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch, 
            train_acc, 
            test_acc,
            total_ce_loss.item() / num_batches,
            total_lb_loss.item() / num_batches,
            total_routing_entropy.item() / num_batches, 
            round(end - start, 3),
            round(max_memory, 1)
        ])


    return train_acc,test_acc


def get_stats(data_loader):
    s = 0
    ss = 0
    num_pixels = 0

    for i, batch in enumerate(data_loader):
        inputs, labels = batch
        num_pixels += inputs.numel()
        s += inputs.sum()
        ss += (inputs ** 2).sum()

    mean = s / num_pixels
    std = ((ss / num_pixels) - (mean ** 2)) ** 0.5

    return mean, std


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a MoE model on Fashion MNIST')

    parser.add_argument('--num_experts', type=int, default=4, help='Number of experts in the MoE model')
    parser.add_argument('--top_k', type=int, default=2, help='Top k experts to route to')
    
    args = parser.parse_args()
    num_experts = args.num_experts
    top_k = args.top_k
    print(f"Number of Experts: {num_experts}, Top K: {top_k}")


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean=(0.2860,), std=(0.3530,))
        ])

    print("Loading Fashion MNIST Data...")
    mnist_trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    mnist_testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    print("Done")

    train_loader = torch.utils.data.DataLoader(
        mnist_trainset,
        batch_size=64,
        shuffle=True,
        num_workers=0
    )

    test_loader = torch.utils.data.DataLoader(
        mnist_testset,
        batch_size=1000,
        shuffle=True,
        num_workers=0
    )

    images, labels = next(iter(train_loader))
    images = sliding_window(images)

    window_size = 5
    height = 28
    width = 28
    pad = 1

    sequence_length = int(((height + 2 * pad) / window_size) ** 2)

    mean, std = get_stats(train_loader)

    learning_rate=0.001
    # momentum=0.9
    max_epochs = 20

    window_size = 5

    model = Model(sequence_length = sequence_length, input_dim = window_size * window_size, hidden_dim = 128, num_classes = 10,
                    input_mean = mean, input_std = std, num_experts=num_experts, top_k=top_k).to(device)   

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = MoELossScheduler(
        total_epochs=20,
        lb_max=0.05,  # start with strong balancing
        lb_min=0.005 # keep a little pressure
    )

    training_stats_filename = "training_stats.csv"
    with open("training_stats.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow([
        "Epoch",
        "Train_Acc",
        "Test_Acc",
        "Average_CE_Loss",
        "Average_LB_Loss",
        "Average_Entropy",
        "Runtime",
        "Memory"
        ])

    for epoch in range(max_epochs):
        train(model, train_loader, test_loader, criterion, optimizer, epoch, device, writer, training_stats_filename, scheduler)

    




    
