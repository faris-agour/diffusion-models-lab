# Diffusion Models Lab  
*A modular laboratory for training, sampling, and visualizing diffusion models — developed during experiments at Samsung Labs.*

> 🔹 **Current model:** DDPM v0.2 (Upgraded UNet + cosine scheduler)  
> 🔹 Previous: DDPM v0.1 (baseline)

---

## 🌫️ Overview

This repository implements a clean, modular, research-oriented framework for **Denoising Diffusion Probabilistic Models (DDPMs)** including:

- Forward (noising) process `q(x_t | x_0)`  
- Reverse (denoising) process `p(x_{t-1} | x_t)`  
- Strong U‑Net backbone with timestep embeddings  
- EMA (Exponential Moving Average) shadow model for high‑quality sampling  
- Checkpointing + resume support  
- Forward & reverse visualization utilities  
- Training and sampling entry scripts  

The architecture follows the core ideas from:

- **Ho et al., DDPM (NeurIPS 2020)**  
- **Song et al., Score-Based Models (ICLR 2021)**  

---

## 🧩 Key Features

### ✔ Full DDPM implementation
- Linear and cosine beta schedules  
- Closed-form forward noising  
- Learned reverse denoising process (ε‑prediction objective)  

### ✔ Strong U‑Net architecture (v0.2)
- Encoder → Bottleneck → Decoder  
- Residual blocks with time conditioning  
- Skip connections at multiple scales  
- Sinusoidal timestep embeddings + MLP  
- Noise prediction head (3‑channel ε̂_θ output)  

### ✔ EMA Model (High-Quality Sampling)
We maintain a shadow model:

```text
EMA = decay * EMA + (1 − decay) * Model
```

EMA improves:
- Sampling sharpness  
- Training stability  
- High-frequency consistency  

Sampling scripts are designed to use the **EMA model checkpoint** when available (e.g., `ddpm_best.pt`).

---

## 🔧 Visualization Tools

This repo includes utilities and notebooks to visualize:

- Progressive noising (forward diffusion)  
- Progressive denoising (reverse diffusion)  
- Forward vs. reverse qualitative behavior  

You can extend `utils.py` and the notebooks in `notebooks/` to generate custom forward/reverse strips and grids.

---

## 📁 Project Structure (DDPM v0.2)

```text
diffusion-models-lab/
│
├── models/
│   ├── model.py          # StrongUNet v2 backbone (time‑conditioned)
│   └── ema.py            # EMA shadow model
│
├── methods/
│   └── ddpm.py           # DDPM: q_sample, p_sample, loss, sampling
│
├── diffusion/
│   └── scheduler.py      # Linear & cosine beta schedules + indexing helpers
│
├── training/
│   ├── train.py          # Training pipeline (optimizer, EMA, checkpoints)
│   └── sample.py         # Sampling script (loads DDPM + checkpoint)
│
├── utils.py              # Visualization helpers (grids, denorm, etc.)
├── data.py               # Stanford Cars dataset loader + transforms
├── notebooks/            # Forward/backward analysis notebooks
├── assets/               # Images & diagrams
├── requirements.txt
└── README.md
```

---

## 🧠 Diffusion Models: Core Idea

Diffusion models learn to **reverse a gradual noise process**.

### 1) Forward (Noising) Process  

We gradually corrupt a clean image \(x_0\):

```math
q(x_t \mid x_0) = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\,\epsilon,\quad \epsilon \sim \mathcal{N}(0, I)
```

After many steps, \(x_T\) becomes nearly pure noise.

### 2) Reverse (Denoising) Process  

The U‑Net learns to predict the noise:

```math
\hat{\epsilon}_\theta(x_t, t)
```

which is then used to compute an estimate of \(x_{t-1}\):

```math
x_{t-1} = f(x_t, \hat{\epsilon}_\theta, t).
```

Starting from random Gaussian noise and iterating this reverse process gives a new generated image.

---

## 🧱 U‑Net Architecture (StrongUNet v2)

### Encoder
- Conv → Norm → SiLU  
- Downsampling via strided convolutions  
- Residual blocks with time embeddings  
- Skip connections stored at each resolution  

### Bottleneck
- Two time‑conditioned residual blocks  
- High‑level structure and global context  

### Decoder
- Transposed conv upsampling  
- Skip concatenation from encoder  
- Residual refinement blocks  

### Timestep Embedding
- Sinusoidal time embedding → MLP  
- Injected into every residual block as an additive conditioning term  

---

## 🔧 Training

From the project root:

```bash
python -m training.train
```

Features:

- DDPM ε‑prediction loss (MSE between true noise and predicted noise)  
- EMA updates during training (shadow model)  
- Checkpoints written under `checkpoints/`, e.g.:  
  - `ddpm_final.pt` – final model  
  - `ddpm_best.pt` – best EMA model (if enabled)  

Resume training from a checkpoint:

```bash
python -m training.train --resume checkpoints/ddpm_best.pt
```

(Adjust the path to match the checkpoint you want to continue from.)

---

## 🎨 Sampling (Using EMA)

Generate images from a trained model:

```bash
python -m training.sample --ckpt checkpoints/ddpm_best.pt
```
Output:

<img width="266" height="266" alt="ddpm_samples_v0 2" src="https://github.com/user-attachments/assets/3ac4f12f-6cae-431b-9ca3-f995118251c7" />

---
Typical behavior:

- Starts from pure Gaussian noise  
- Runs the reverse DDPM process for `T` steps (e.g., 400)  
- Saves a grid under `assets/` (e.g., `ddpm_samples_epoch50.png`)  

You can customize batch size, image size, and output path inside `training/sample.py`.

---

## 🔍 Forward / Reverse Analysis

For deeper inspection of the diffusion process, use the notebooks in `notebooks/`:

- `01_forward_diffusion.ipynb` — visualize how a clean image is gradually noised  
- `02_backward_diffusion.ipynb` — explore denoising behavior and sampling trajectories  

You can combine these with helpers from `utils.py` to export forward/reverse strips for reports or slides.

---

## 📌 Versioning

This repo uses **simple version tags** on Git to track model evolution.

| Version Tag | Model Type | Description |
|-------------|------------|-------------|
| **v0.2**    | DDPM‑v2    | Current version – StrongUNet v2 + cosine scheduler + cleaned repo structure |
| **v0.1**    | DDPM‑v1    | Initial baseline implementation (simpler UNet, linear schedule) |

Tagging the current state as **v0.2**:

```bash
git tag v0.2
git push origin v0.2
```

(You can also create a GitHub Release pointing to this tag.)

To reproduce or inspect the older baseline (v0.1), check out the `v0.1` tag once it is created:

```bash
git checkout v0.1
```

---

## 🔬 Tips & Next Steps

- Use EMA checkpoints (`ddpm_best.pt`) for the best visual quality.  
- Higher resolutions require increasing UNet capacity (more channels, attention).  
- Cosine beta schedules generally outperform purely linear schedules.  
- Sampling speed can be improved by adding **DDIM** or other fast samplers on top of this DDPM core.  
- Future versions of this lab can include:
  - DDPM‑Improved (attention UNet + advanced schedules)  
  - DDIM (fast deterministic sampling)  
  - Latent Diffusion (diffusion in VAE latent space)  

---

## 🧪 Using the Trained Model in Your Own Code

You can also load the trained model and DDPM wrapper directly in your own scripts, for example in a notebook or another project:

```python
import torch

from models.model import StrongUNet
from methods.ddpm import DDPM

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1) Restore UNet
model = StrongUNet(img_ch=3, time_dim=256).to(device)

ckpt = torch.load("checkpoints/ddpm_best.pt", map_location=device)
state_dict = ckpt.get("model", ckpt.get("ema_model", ckpt))
model.load_state_dict(state_dict)
model.eval()

# 2) Wrap with DDPM
ddpm = DDPM(
    model=model,
    timesteps=400,
    beta_schedule="cosine",
    device=device,
)

# 3) Sample a small batch
with torch.no_grad():
    samples = ddpm.sample(batch_size=4, img_channels=3, img_size=64)  # [-1, 1]
```

You can then denormalize and visualize `samples` using helpers from `utils.py` or your own plotting code.

---

## 📚 References

- Ho et al., *Denoising Diffusion Probabilistic Models* (NeurIPS 2020)  
- Song et al., *Score-Based Generative Modeling* (ICLR 2021)  
- Karras et al., *Elucidating Diffusion Models* (CVPR 2022)  
- Koppers et al., *Diffusion Models for Medical Imaging* (MedIA 2023)  
