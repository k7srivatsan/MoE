import matplotlib.pyplot as plt 
import numpy as np

from collections import defaultdict

import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm 

from losses.MoE_loss import MoELoss, MoELossOutput

from skimage.util import view_as_windows

from Model import Model

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
                draw(images, expert_positions, epoch)
                first = False
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
def draw(images, expert_positions, epoch):
    images = images.view(-1, images.shape[2])
    print(images.shape)

    num_experts = len(expert_positions.keys())

    fig, ax = plt.subplots(num_experts)

    for expert in range(num_experts):
        token_positions = torch.cat(expert_positions[expert], dim = 0)
        if(token_positions.allclose(torch.zeros_like(token_positions))):
            print("zeros")
            tokens = torch.zeros(5, 5)
        else:    
            print(token_positions)
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

def train(model, train_loader, test_loader, criterion, optimizer, epoch, device):

    model.train()
    running_loss = 0

    with tqdm.tqdm(train_loader, total=len(train_loader)) as tepoch:
        for i, data in enumerate(tepoch):
            inputs, labels = data

            inputs = sliding_window(inputs)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, moe_outputs = model(inputs)
            ce_loss = criterion(outputs, labels)
            # load_balance = moe_outputs.balance
            lb_loss = moe_outputs.lb_loss
            routing_entropy =  moe_outputs.routing_entropy 

            final_loss = ce_loss

            tepoch.set_description(f"Epoch {epoch}")
            tepoch.set_postfix(loss=final_loss.item())

            final_loss.backward()
            optimizer.step()

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
    max_epochs = 25

    window_size = 5

    model = Model(sequence_length = sequence_length, input_dim = window_size * window_size, hidden_dim = 128, num_classes = 10,
                    input_mean = mean, input_std = std).to(device)   

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(device)

    for epoch in range(max_epochs):
        train(model, train_loader, test_loader, criterion, optimizer, epoch, device)

    




    
