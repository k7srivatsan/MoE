# Patch-wise Mixture of Experts (MoE) on FashionMNIST

A minimal Mixture of Experts (MoE) model built from scratch in PyTorch, using patch-wise routing over FashionMNIST. This project explores sparse expert activation, routing specialization, and tradeoffs in compute, accuracy, and interpretability.

---

## Project Highlights

- Implements a custom **top-k MoE** architecture
- Uses **5×5 image patches** as routing tokens
- Trains with **soft routing**, evaluates with **top-k expert selection**
- Visualizes per-expert token prototypes and specialization
- Tracks CE loss, load balancing loss, routing entropy
- Logs runtime, memory, and routing behavior over epochs
- Easily modifiable for speech tokens

---

## Model Overview

- **Input**: FashionMNIST image → patchify into 25–36 tokens (5×5 patches)
- **Patch encoder**: 2-layer MLP (`25 → 64 → D`)
- **Router**: Softmax-based token-wise gating (`D → num_experts`)
- **Experts**: MLPs (`D → D → D`) with ReLU + Dropout
- **Classifier**: MLP over pooled expert outputs

```text
            +-----------------------------+
Image --->  | Patchify → Patch Encoder    |  → Tokens (T, D)
            +-----------------------------+
                             ↓
                   +-----------------+
                   |   MoE Router    |  → Softmax over E
                   +-----------------+
                            ↓
        +--------------------+--------------------+
        | Expert 0 (MLP)     | ...   Expert N (MLP)|
        +--------------------+--------------------+
                            ↓
             Token Outputs (sparse or weighted)
                            ↓
              Mean Pool → Classifier MLP → Prediction
```
Mean 5×5 patch routed to each expert (epoch 24)
<p align="center"> <img src="expert_tokens_epoch_4.png" width="300"/> </p>
Routing heatmap: which expert each patch in the image was routed to
<p align="center"> <img src="MoE_epoch_4.png" width="300"/> </p>

---

## Running Model

To run the model, simply type ```python train_model.py``` with optional arguments:
- --num_experts
- --soft_train  
  Controls whether to use top-k selection during training.  
  Set to `true` to use **all** experts during training (default: `true`).
- --top_k
- --max_epochs
