# Diffusion Models

> A lab for experiments and implementations of **diffusion models** in AI provided by SAMSUNG Labs.

---

## 🌫️ What Are Diffusion Models?

Diffusion models are a class of *generative models* that learn to create data by **reversing a gradual noising process**.  
They start from pure noise and iteratively **denoise** it to produce realistic data (images, signals, even 3D volumes).

Unlike GANs (which try to map noise → data in one shot), diffusion models learn **a sequence of small, stable steps** that model the data distribution with high precision.

---



## 🧠 Step 1 — Forward (Noising) Process

**Goal:** Show how a clean image becomes progressively noisier as the timestep `t` increases from `0` to `T-1`.

**What happens conceptually**  
- We pick a clean image \(x_0\).  
- For a set of timesteps (e.g., 10 evenly spaced points between `0` and `T-1`), we compute noisy versions \(x_t\) using the closed-form forward diffusion equation.  
- Each \(x_t\) blends the clean image with Gaussian noise according to a noise schedule.  
- The result is a left‑to‑right “strip” where structure fades and noise dominates as `t` grows.

**How to reproduce (high level)**  
1. **Select a representative image** from your dataloader (ideally one with color to avoid grayscale demos).  
2. **Choose a list of timesteps** across your configured `T` (e.g., `10–12` points spaced evenly).  
3. **Compute the noisy images** at each timestep using your forward diffusion function.  
4. **Render a visual strip**: place the results side‑by‑side with labels like `t=0, t=30, …, t=T-1`.  
5. **Export at a high resolution** (e.g., 300 DPI) so it looks crisp in the README.

> **Note:** You are free to set `T` (e.g., 100 / 300 / 1000). If you change `T`, recompute the schedule and keep any selected `t` in `[0, T-1]`.

---


## 📷 Forward Process (Placeholder)

<!-- Place your saved image here (created by `save_forward_strip`) -->
<p align="center">
  <img width="1172" height="142" alt="image" src="https://github.com/user-attachments/assets/2db10fcd-6afd-4d3f-a6f0-ca8c8b3316cd" />

</p>

---

## ⏪ Step 2 — Reverse (Denoising) Process & U-Net Backbone

Once the forward process has gradually corrupted a clean image \(x_0\) into nearly pure noise \(x_T\),  
the **reverse diffusion process** learns to invert this transformation step by step.

Conceptually, the model:

- Starts from pure noise \(x_T \sim \mathcal{N}(0, I)\).
- At each timestep \(t = T-1, \dots, 0\), predicts the noise component in the current image.
- Uses this prediction to update \(x_t \rightarrow x_{t-1}\) with a small denoising step.
- After all steps, returns a clean (or at least realistic) sample \(x_0\).

### 🔍 What the Model Learns

Instead of directly predicting the clean image, the network is trained to **predict the added noise**:

- During training:
  - Sample a real image \(x_0\).
  - Sample a timestep \(t\).
  - Generate a noisy version \(x_t\) using the same closed-form forward process as before.
  - Train the network to predict the noise \(\hat{\epsilon}_\theta(x_t, t)\).
  - Use a simple MSE loss between the true noise and the predicted noise.

This formulation is stable, simple, and works extremely well in practice.

---

## 🧱 U-Net Architecture (Backbone)

We use a **U-Net** as the backbone for the denoising model:

- **Encoder (down path)**  
  - Repeated blocks of:  
    - 2D convolutions  
    - Normalization + non-linearity  
    - Downsampling (reducing spatial resolution, increasing channels)
  - Each block saves a **skip connection** feature map for the decoder.

- **Bottleneck (middle)**  
  - One or more residual blocks operating at the lowest resolution.
  - Captures global structure and long-range context.

- **Decoder (up path)**  
  - Mirrors the encoder:
    - Upsampling (transpose conv or interpolation + conv)
    - Concatenation with corresponding encoder features (skip connections)
    - Residual blocks to refine details
  - Gradually reconstructs spatial detail as resolution increases.

- **Output head**  
  - A final 1×1 convolution maps the last feature map back to the same number of channels as the input image  
    (e.g. 3 channels for RGB noise prediction).

### Key Design Choices

- **Input / output:**  
  - Input: \(x_t\) in \([-1, 1]\), shape \([B, 3, H, W]\).  
  - Output: predicted noise \(\hat{\epsilon}_\theta\) with the same shape.

- **Depth & channels:**  
  - Multiple resolution levels (downsampling → upsampling).  
  - Channels typically grow as resolution shrinks (e.g. 64 → 128 → 256 → 512 → 1024).

- **Skip connections:**  
  - Preserve high-frequency detail by forwarding encoder features directly to matching decoder levels.

---

## ⏱️ Timestep Encoding (Time Embedding)

The model also needs to know **which diffusion step** it is currently denoising.

We therefore encode the scalar timestep \(t\) into a **time embedding vector**:

- A sinusoidal (Fourier-style) embedding maps \(t\) to a vector of fixed size (the time embedding dimension).
- This vector is then passed through a small MLP to obtain a richer representation.
- The time embedding is injected into the U-Net blocks (e.g. as an additive bias on the feature channels).

Intuitively:

- Early timesteps (high noise) → the network focuses on coarse, global structure.  
- Late timesteps (low noise) → the network refines fine-grained details and textures.

---

## 🧩 Putting It Together

1. **Forward process**:  
   - Defines how to corrupt an image \(x_0\) into \(x_t\) with a chosen noise schedule.

2. **Reverse model (U-Net + time embedding)**:  
   - Learns to predict the noise \(\hat{\epsilon}_\theta(x_t, t)\) at each step.

3. **Sampling loop**:  
   - Starts from pure noise \(x_T\).  
   - Iteratively applies denoising steps using the model’s predictions until reaching \(x_0\).

This backbone is flexible: it can be extended with conditioning (e.g. text, masks, diffusion MRI signals) or modified to follow alternative parameterizations (e.g. predicting \(x_0\) or v-parameterization) without changing the core idea.

---
## ⚖️ Diffusion vs. Traditional Generative Models

| Feature | Diffusion Models (DMs) | Traditional Generative AI (GANs / VAEs) |
|---|---|---|
| **Core idea** | Gradually remove noise (reverse diffusion) | Directly map noise → data |
| **Training stability** | Very stable, predictable loss | Often unstable, needs tricks |
| **Sample diversity** | High (less mode collapse) | May collapse to few modes |
| **Output quality** | Extremely detailed and realistic | Good, but may show artifacts |
| **Computation** | Slower (many denoising steps) | Faster (one forward pass) |
| **Control / conditioning** | Easy to integrate (text, mask, etc.) | Harder to control precisely |

---


## 🧬 Why Diffusion Models Matter in dMRI

In diffusion MRI (dMRI), we measure how water molecules **diffuse** in tissue, giving insight into microstructure and white-matter pathways.  
Diffusion models in AI share a **conceptual link**: they model *probabilistic diffusion* in feature space.

Applying diffusion models to dMRI can:

- **Denoise** raw diffusion data while preserving microstructural detail.  
- **Generate synthetic diffusion signals** to augment datasets or test tractography algorithms.  
- **Reconstruct missing gradients or directions** in under-sampled acquisitions.  
- **Model uncertainty** in fiber orientation distributions in a principled way.

> Diffusion models provide a mathematically consistent, noise-aware framework — a great fit for the stochastic nature of diffusion MRI.

---

## 🧪 Notes & Tips

- Use **cosine** or **linear** schedules; cosine often yields smoother noising:
  ```python
  import math, torch

  def cosine_beta_schedule(timesteps, s=0.008):
      steps = timesteps
      x = torch.linspace(0, steps, steps+1, dtype=torch.float32)
      alphas_cum = torch.cos(((x/steps)+s)/(1+s) * math.pi/2) ** 2
      alphas_cum = alphas_cum / alphas_cum[0]
      betas = 1 - (alphas_cum[1:] / alphas_cum[:-1])
      return betas.clamp(1e-5, 0.999)
  ```
- For clean visualizations in README, use higher `dpi` (e.g., 300) and larger `scale`.
- Some dataset images may be truly **grayscale**; that’s expected. The noising can look “colored” if noise is sampled independently per channel.

---

## 📚 References

- Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020  
- Song et al., *Score-Based Generative Modeling through Stochastic Differential Equations*, ICLR 2021  
- Koppers et al., *Diffusion Models for Medical Imaging*, MedIA 2023  

---

> “From noise comes structure — both in the brain and in generative modeling.”
