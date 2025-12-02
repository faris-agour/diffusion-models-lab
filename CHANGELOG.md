v0.2 – DDPM Upgrade Release

Release date: 2025-12-02

🔧 Major Architecture Fixes

Replaced the old StrongUNet with StrongUNet v2

Fixed all channel-mismatch errors

Added correct UpBlock(in_ch, skip_ch, out_ch) design

Skip-connection dimensions are now fully consistent

Improved bottleneck and decoder structure

Better residual blocks + GroupNorm stability

🕒 Better Timestep Embedding

Added stable sinusoidal time embedding

Improved conditioning via a deeper MLP head

More expressive diffusion step injection

🧹 Cleaner Repository Structure

Removed unused/experimental files

Final structure:

models/
methods/
diffusion/
training/
notebooks/
assets/


Deleted duplicated utils + removed legacy notebooks

🎯 Improved Sampling Reliability

Now uses correct EMA model checkpoint by default

Sampling loop updated to match U-Net v2

Fixes silent shape errors during sampling

📦 Versioning Enabled

Added project versioning (v0.2) via Git tags

Repo now supports multiple diffusion model versions (DDPM first)

v0.1 – Initial DDPM Implementation

Basic linear-schedule DDPM

Early U-Net architecture (minimal)

Working training + sampling scripts

Initial visualization / forward process notebooks

First working end-to-end DDPM pipeline

🎯 Now your README should include this:

At the END of your README, add:

📌 Version History
v0.2 – DDPM Upgrade

Major U-Net v2 redesign

Fixed channel mismatches

Cleaned project structure

Stable sinusoidal time embeddings

EMA-based sampling improvements

Ready for next diffusion models (DDIM, DDPM++, etc.)

v0.1 – First Release

Initial DDPM implementation

Basic training loop, sampler, visualization