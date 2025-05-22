# Patch-wise Mixture of Experts (MoE) on FashionMNIST

A minimal Mixture of Experts (MoE) model built from scratch in PyTorch, using patch-wise routing over FashionMNIST. This project explores sparse expert activation, routing specialization, and tradeoffs in compute, accuracy, and interpretability.

---

## 🔧 Project Highlights

- ✅ Implements a custom **top-k MoE** architecture without HuggingFace
- ✅ Uses **5×5 or 6×6 non-overlapping image patches** as routing tokens
- ✅ Trains with **soft routing**, evaluates with **top-k expert selection**
- ✅ Visualizes per-expert token prototypes and specialization
- ✅ Tracks CE loss, load balancing loss, routing entropy
- ✅ Logs runtime, memory, and routing behavior over epochs

---

## 🧠 Model Overview

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
