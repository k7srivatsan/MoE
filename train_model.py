import matplotlib.pyplot as plt 

from collections import defaultdict

import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm 

from losses.MoE_loss import MoELoss, MoELossOutput

from Model import Model

def accuracy(model, loader, device, epoch):
    correct = 0
    total = 0
    with torch.no_grad():
        first = True
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            images = sliding_window(images)
            outputs, moe_outputs = model(images)
            if first:
                expert_tokens = moe_outputs.expert_tokens
                draw(expert_tokens, epoch)
                first = False
            probs = nn.functional.softmax(outputs, dim = -1)
            pred_labels = torch.argmax(probs, dim = -1)
            correct += (pred_labels == labels).sum().item()
            total += torch.numel(labels)

    accuracy = (correct / total) * 100
    return accuracy


def sliding_window(inputs, window_size = 5):
    inputs = inputs.squeeze(1)
    B, T, C = inputs.shape

    tokens = []

    for i in range(T - window_size + 1):
        window = inputs[:, i:i + window_size, :] # B x window_size x C
        window = window.reshape(B, window_size * C) 
        tokens.append(window)

    inputs = torch.stack(tokens, dim = 1)
    return inputs

# explainability
def draw(expert_tokens, epoch):
    num_experts = len(expert_tokens.keys())

    fig, ax = plt.subplots(num_experts)

    for expert in range(num_experts):
        tokens = expert_tokens[expert]
        tokens = torch.cat(tokens, dim = 0)
        tokens = tokens.detach().cpu()
        # tokens = tokens * self.input_std + self.input_mean
        tokens = torch.mean(tokens, dim = 0)
        tokens = tokens.reshape(5, 28)

        ax[expert].imshow(tokens, cmap='gray', vmin = 0, vmax = 1)
        ax[expert].set_title(f"Expert {expert}'s Tokens")

    fig.tight_layout()
    plt.savefig(f"specializations/expert_tokens_epoch_{epoch}.png")
    plt.close()
    print("Drew Tokens!")   

def train(model, train_loader, test_loader, criterion, optimizer, epoch, device):
    model.reset_histogram(device)

    model.train()
    running_loss = 0

    with tqdm.tqdm(train_loader, total=len(train_loader)) as tepoch:
        for i, data in enumerate(tepoch):
            inputs, labels = data

            inputs, labels = inputs.to(device), labels.to(device)
            inputs = sliding_window(inputs)
            optimizer.zero_grad()
            outputs, moe_outputs = model(inputs)
            ce_loss = criterion(outputs, labels)
            lb_loss = moe_outputs.lb_loss

            final_loss = ce_loss + 0.01 * lb_loss

            tepoch.set_description(f"Epoch {epoch}")
            tepoch.set_postfix(loss=final_loss.item())

            final_loss.backward()
            optimizer.step()

            # print statistics
            running_loss += final_loss.item()
            if i % 200==199:
                load_balance = model.get_load_balance()
                print(load_balance)
                print("[epoch %d, iter %5d] loss: %.3f"%(epoch,i+1,running_loss/200))
                running_loss=0.0

    model.save_histogram(epoch)
    train_acc = accuracy(model, train_loader, device, epoch)
    test_acc = accuracy(model, test_loader, device, epoch)
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
        shuffle=False,
        num_workers=0
    )

    mean, std = get_stats(train_loader)

    learning_rate=0.001
    # momentum=0.9
    max_epochs = 25

    window_size = 5

    model = Model(sequence_length = 28 - window_size + 1, input_dim = 28 * window_size, hidden_dim = 128, num_classes = 10,
                    input_mean = mean, input_std = std).to(device)   

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(device)

    for epoch in range(max_epochs):
        train(model, train_loader, test_loader, criterion, optimizer, epoch, device)

    




    
