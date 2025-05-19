from collections import defaultdict
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torchvision

import math


class ExpertMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ExpertMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class MoE(nn.Module):
    def __init__(self, sequence_length, input_dim, hidden_dim, num_experts, k, log_usage = True):
        super(MoE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim 
        self.num_experts = num_experts
        self.sequence_length = sequence_length
        self.k = k
        self.log_usage = log_usage

        # input (B, T, C)
        # if non-tokenized data, T dimension is just 

        # router routes to experts based on input channel characteristics
        self.router = nn.Linear(input_dim, num_experts)    

        self.experts = nn.ModuleList([ExpertMLP(self.input_dim, self.hidden_dim) for _ in range(self.num_experts)])

        self.sequence_histogram = torch.zeros(sequence_length, num_experts)

        self.return_lb_loss = True

        self.f = None

    # def positional_encoding():
    #     # positional embedding to add to x

    def get_load_balance(self):
        return self.f
        
    def forward(self, x):
        B, T, C = x.shape

        logits = self.router(x)
        clean_logits = logits
        
        # if epoch != -1:
        #     std = 0.01 * math.exp(-0.5 * epoch)
        #     # 0.01e^{-0.5x}
        #     logits += torch.randn_like(logits) * std # to encourage exploration


        topk_vals, topk_indices = torch.topk(logits, self.k, dim = -1) # (B, T, k experts)

        # lb_loss = self.compute_lb_loss(clean_logits, topk_indices)

        # if epoch != -1: 
        #     temperature = 1.0 * math.exp(-0.05 * epoch)
        #     topk_vals = topk_vals / temperature

        if self.log_usage:
            with torch.no_grad():
                for t in range(T):
                    # t'th point in seqeunce, get expert histogram across batch
                    t_indices = topk_indices[:, t, :]
                    self.sequence_histogram[t] += torch.bincount(t_indices.flatten(), minlength = self.num_experts)


        router_scores = nn.functional.softmax(topk_vals, dim = -1) # (B, t, k experts)

        x_flat = x.view(-1, x.shape[-1]) # (B x T, C)
        topk_indices_flat = topk_indices.view(-1, topk_indices.shape[-1]) # (B x T, k)
        router_scores_flat = router_scores.view(-1, router_scores.shape[-1]) # (B x T, k)

        expert_tokens = defaultdict(list)
        expert_weights = defaultdict(list)
        expert_positions = defaultdict(list)
        
        for i in range(self.k):
            ith_indices = topk_indices_flat[:, i]
            ith_router_score = router_scores_flat[:, i]
            
            for expert in range(self.num_experts):
                mask = (ith_indices == expert) # 1 where token is redirected to expert
                
                if mask.any():
                    expert_tokens[expert].append(x_flat[mask]) # all the tokens redirected to this expert
                    expert_weights[expert].append(ith_router_score[mask])
                    expert_positions[expert].append(mask.nonzero(as_tuple=False).view(-1))

        # TODO: send it through experts and aggregate

        output = torch.full_like(x_flat, 0)

        for expert in range(self.num_experts):  

            #some expert may never be chosen 
            if(len(expert_tokens[expert]) == 0):
                continue

            tokens = torch.cat(expert_tokens[expert], dim = 0)
            weights = torch.cat(expert_weights[expert], dim = 0)
            positions = torch.cat(expert_positions[expert], dim = 0)

            expert_output = self.experts[expert](tokens) # (samples, input_dim)
            # weights is (samples, 1)
            output[positions] += torch.einsum("ij,i->ij", expert_output, weights)

        output = output.reshape((B, T, C))
        return clean_logits, topk_indices, expert_tokens, output

    def save_histogram(self, epoch):
        fig, ax = plt.subplots()

        normalized_histogram = self.sequence_histogram / torch.sum(self.sequence_histogram, dim=1).unsqueeze(1)
        normalized_histogram = normalized_histogram.detach().to('cpu')

        pos = ax.imshow(normalized_histogram, cmap='viridis')

        ax.set_title("Experts chosen at each index in sequence")
        ax.set_ylabel("Sequence Index")
        ax.set_xlabel("Expert Distribution")

        fig.colorbar(pos, ax = ax)
        plt.savefig(f"histograms/MoE_epoch_{epoch}.png")
        plt.close()

    def reset_histogram(self, device):
        self.sequence_histogram = self.sequence_histogram.to(device)
        self.sequence_histogram.zero_()


if __name__ == "__main__":

    MixtureModel = MoE(sequence_length = 24, input_dim=140, hidden_dim=128, num_experts=4, k = 2)

    x_test = torch.zeros(32, 24, 140)
    # x_output, lb_loss = MixtureModel(x_test, epoch = -1, draw = True)
    # print(lb_loss)

    expert_tokens = defaultdict(list)
    expert_tokens[0] = [torch.randn(100, 140) * 0.3081 + 0.1307]
    expert_tokens[1] = [torch.randn(400, 140) * 0.3081 + 0.1307]
    expert_tokens[2] = [torch.randn(100, 140) * 0.3081 + 0.1307]
    expert_tokens[3] = [torch.randn(168, 140) * 0.3081 + 0.1307]
    MixtureModel.draw(expert_tokens)



















