
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
    expert_tokens: Tensor # used for drawing
    output: Tensor
    lb_loss: float
    routing_entropy: float


class MoELoss(nn.Module):

    def __init__(self, moe: MoE) -> None:
        super().__init__()
        self.moe = moe

    def compute_lb_loss(self, clean_logits, topk_indices):
        B, T, K = topk_indices.shape
        
        probs = nn.functional.softmax(clean_logits, dim = -1)
        avg_probs = probs.mean(dim=(0, 1)) # dim (self.num_experts,)

        f = torch.zeros(self.moe.num_experts, device = topk_indices.device)

        one_hot = nn.functional.one_hot(topk_indices.flatten(), num_classes = self.moe.num_experts)
        f += one_hot.sum(dim = 0)
        
        f /= (B * T * K)

        self.f = f

        return self.moe.num_experts * torch.dot(f, avg_probs)


    def compute_routing_entropy(self):
        return 0


    def forward(self, tokens):

        clean_logits, topk_indices, expert_tokens, output = self.moe(tokens)

        lb_loss = self.compute_lb_loss(clean_logits, topk_indices)
        routing_entropy = self.compute_routing_entropy()

        return MoELossOutput(clean_logits, topk_indices, expert_tokens, output, lb_loss, routing_entropy)







