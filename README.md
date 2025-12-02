
# Diffusion Models Lab  
*A modular laboratory for training, sampling, and visualizing diffusion models — developed for SAMSUNG Labs.*

---

# 🌫️ Overview

This repository implements a clean, modular, research-oriented framework for **Denoising Diffusion Probabilistic Models (DDPMs)** including:

- Forward (noising) process `q(x_t | x_0)`
- Reverse (denoising) process `p(x_{t-1} | x_t)`
- Strong U‑Net backbone with timestep embeddings
- EMA (Exponential Moving Average) shadow model for high‑quality sampling
- Checkpointing + resume support
- Forward & reverse visualization utilities
- Training and sampling entry scripts


---

# 🧩 Key Features

### ✔ Full DDPM implementation
- Linear beta schedule
- Closed-form forward noising
- Learned reverse denoising process

### ✔ Strong U-Net architecture
- Encoder → Bottleneck → Decoder
- Residual blocks
- Skip connections
- Timestep sinusoidal embeddings
- Noise prediction head (3-channel ε̂θ output)

### ✔ EMA Model (High-Quality Sampling)
We maintain a shadow model:

```
EMA = decay * EMA + (1−decay) * Model
```

EMA drastically improves:
- Sampling sharpness  
- Stability  
- High-frequency consistency  

Sampling now uses the **EMA model by default**.

## 🔧 Visualization Tools

This repo includes utilities to visualize:

- Progressive noising (forward diffusion)
- Progressive denoising (reverse diffusion)
- Full side‑by‑side forward/reverse grids

---
## 📁 Project Structure (DDPM v1)

```
diffusion-models-lab/
│
├── models/
│   ├── model.py                 # Strong UNet backbone
│   └── ema.py                   # EMA shadow model
│
├── methods/
│   └── ddpm.py                  # DDPM: q_sample, p_sample, loss, sampling
│
├── diffusion/
│   └── scheduler.py             # Linear & cosine beta schedules
│
├── training/
│   ├── train.py                 # Training pipeline (EMA, checkpoints)
│   └── sample.py                # Sampling script (EMA-enabled)
│
├── utils.py                     # Visualization utilities
├── data.py                      # Stanford Cars dataset loader
├── notebooks/                   # Forward/backward analysis notebooks
├── assets/                      # Images & diagrams
├── requirements.txt
└── README.md
```

---

# 🧠 Diffusion Models: Core Idea

Diffusion models learn to **reverse a noise process**.

### 1) Forward (Noising) Process  
We gradually corrupt a clean image \(x_0\)


After many steps, \(x_T\) becomes nearly pure noise.

### 2) Reverse (Denoising) Process  
The U-Net learns:

\[
\epsilon_	heta(x_t, t)
\]

which is used to compute:

\[
x_{t-1} = f(x_t, \hat{\epsilon}_	heta)
\]

This produces a new sample starting from random noise.

---

# 🌫️ Forward Diffusion Visualization

This strip shows how a clean image becomes increasingly noisy.

<p align="center">
  <img src="https://github.com/user-attachments/assets/2db10fcd-6afd-4d3f-a6f0-ca8c8b3316cd"/>
</p>

---

# ⏪ Reverse Denoising Visualization (Using EMA)

The reverse process starts from pure noise and progressively reconstructs structure:


---

# 🧱 U-Net Architecture

### Encoder
- Conv → Norm → Activation  
- Downsampling  
- Save skip connections  

### Bottleneck
- Residual blocks  
- Global structure modeling  

### Decoder
- Upsampling  
- Skip concatenation  
- Residual refinement  

### Timestep embedding
We encode the diffusion step using sinusoidal embeddings + MLP injection.

---

# 🔧 Training

Run training:

```
python -m training.train   
```

Features:
- EMA updates every step  
- Checkpoints every 5 epochs  
- Best model saved automatically

Resume training:

```
python -m training.train --resume checkpoints/epoch_20.pt
```

---

## 🎨 Sampling (Using EMA)

Generate images:

```
python -m training.sample --ckpt checkpoints/best_model.pt
```

Output:

<img src="https://github.com/user-attachments/assets/88dd95be-888f-444b-9f24-1f8a2212438c"
     alt="sample"
     width="400">



Uses:
- EMA weights 
- 200-step denoising loop  
- Strong U-Net backbone  

---

# 🔍 Forward + Reverse Visualization

To generate BOTH forward & reverse strips:

```
python visualize_forward_reverse.py --ckpt checkpoints/best_model.pt
```

Output:

<img src="https://github.com/user-attachments/assets/c1d7c798-e6b0-4866-875b-1ee9105f06b8"
     alt="reverse"
     width="800">




---

## 📌 Versioning (IMPORTANT)

This repo uses **semantic versioning for models**, not code only.

| Version Tag | Model Type | Description |
|-------------|------------|-------------|
| **v1.0** | DDPM‑v1 | Current version (you are here) |
| **v2.0** | DDPM‑Improved | UNet‑v2 + Attention + Cosine schedule + EMA‑optimized |

### To tag this version as DDPM‑v1:

```
git tag v1.0
git push origin v1.0
```

---

# 🔬 Tips

- Use EMA for best samples  
- Higher resolution needs bigger U-Net  
- Cosine schedules often outperform linear  
- Sampling speed can be improved with DDIM  

---
