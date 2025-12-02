# 📌 Version History

## v0.2 – DDPM Upgrade Release
**Release date:** 2025-12-02

### 🔧 Major Architecture Fixes
- Replaced the old **StrongUNet** with **StrongUNet v2**
- Fixed all **channel-mismatch** errors across the network
- Added the correct **`UpBlock(in_ch, skip_ch, out_ch)`** interface
- Skip-connection dimensions are now **fully consistent** end-to-end
- Improved **bottleneck** and **decoder** structure
- Added stronger **residual blocks** with **GroupNorm** for stability

### 🕒 Better Timestep Embedding
- Added a stable **sinusoidal timestep embedding**
- Improved conditioning via a **deeper MLP head**
- More expressive diffusion-step injection throughout the network

### 🧹 Cleaner Repository Structure
- Removed unused and experimental files
- Deleted duplicated utilities and removed legacy notebooks
- Final structure:

```text
models/
methods/
diffusion/
training/
notebooks/
assets/
```

### 🎯 Improved Sampling Reliability
- Sampling now uses the **EMA model checkpoint** by default
- Sampling loop updated to match **StrongUNet v2**
- Fixes silent shape issues during sampling

### 📦 Versioning Enabled
- Added project versioning via **Git tags** (`v0.2`)
- Repository layout now supports **multiple diffusion model versions**
  - **DDPM** first, with a clean path for **DDIM / DDPM++ / others**

---

## v0.1 – Initial DDPM Implementation

### Highlights
- Basic **linear-schedule DDPM**
- Early U-Net architecture (minimal but functional)
- Working **training** and **sampling** scripts
- Initial visualization / forward-process notebooks
- First working end-to-end DDPM pipeline
