# Diffusion Models

> A lab for experiments and implementations of **diffusion models** in AI provided by SAMSUNG Labs.

---

## 🌫️ What Are Diffusion Models?

Diffusion models are a class of *generative models* that learn to create data by **reversing a gradual noising process**.  
They start from pure noise and iteratively **denoise** it to produce realistic data { images, signals, or even 3D volumes }.

Instead of learning to generate samples in one shot (like GANs), they learn **a sequence of small, stable steps** that model the data distribution with high precision.

---

## ⚖️ Diffusion vs. Traditional Generative Models

| Feature | Diffusion Models (DMs) | Traditional Generative AI (GANs / VAEs) |
|----------|------------------------|-----------------------------------------|
| **Core idea** | Gradually remove noise (reverse diffusion) | Directly map noise → data |
| **Training stability** | Very stable, predictable loss | Often unstable, needs tricks (e.g. GAN loss balance) |
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

In short:  
> Diffusion models provide a mathematically consistent, noise-aware framework — a perfect fit for the stochastic nature of diffusion MRI.

---

## 📚 References

- Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020  
- Song et al., *Score-Based Generative Modeling through Stochastic Differential Equations*, ICLR 2021  
- Koppers et al., *Diffusion Models for Medical Imaging*, MedIA 2023  

---

> “From noise comes structure —> both in the brain and in generative modeling.”
