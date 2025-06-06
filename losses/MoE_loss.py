
import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F

from MoE import MoE

from dataclasses import dataclass


@dataclass
class MoELossOutput:
    clean_logits: Tensor # both used for lb_loss
    topk_indices: Tensor 
    expert_positions: Tensor # used for drawing
    output: Tensor
    load_balance: Tensor
    lb_loss: float
    routing_entropy: float


class MoELoss(nn.Module):

    def __init__(self, moe: MoE) -> None:
        super().__init__()
        self.moe = moe

    def compute_lb_loss(self, clean_logits, topk_indices):
        B, T, K = topk_indices.shape
        
        probs = nn.functional.softmax(clean_logits, dim = -1)
        # avg_probs = probs.mean(dim=(0, 1)) # dim (self.num_experts,)

        load_balance = torch.zeros(self.moe.num_experts, device = topk_indices.device)

        one_hot = nn.functional.one_hot(topk_indices.flatten(), num_classes = self.moe.num_experts)
        load_balance += one_hot.sum(dim = 0)
        
        load_balance /= (B * T * K)

        avg_probs = probs.mean(dim=(0, 1))
        avg_probs = avg_probs / avg_probs.sum()

        uniform = torch.full_like(avg_probs, 1.0 / avg_probs.numel())
        lb_loss = F.kl_div(avg_probs.log(), uniform, reduction="sum")

        return load_balance, lb_loss
        # self.moe.num_experts * torch.dot(f, avg_probs)


    def compute_routing_entropy(self, clean_logits):
        B, T, C = clean_logits.shape
        probs = F.softmax(clean_logits, dim = -1)
        entropy = torch.sum(probs * -torch.log(probs), dim = -1)
        entropy = torch.mean(entropy)
        return entropy

    def forward(self, tokens):
        clean_logits, topk_indices, expert_positions, output = self.moe(tokens)

        load_balance, lb_loss = self.compute_lb_loss(clean_logits, topk_indices)
        routing_entropy = self.compute_routing_entropy(clean_logits)

        return MoELossOutput(clean_logits, topk_indices, expert_positions, output, load_balance, lb_loss, routing_entropy)







