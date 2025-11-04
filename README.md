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
