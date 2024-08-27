# Stable Diffusion Negative Prompt Reformulation

This repository explores and proposes a reformulation of the negative prompt mechanism used in Stable Diffusion. The goal is to refine the existing configuration, leading to improved outcomes in generated images when using both positive and negative prompts.

## Overview

### Classifier-Free Guidance (CFG) Formulation

The Classifier-Free Guidance (CFG) used in diffusion models, including Stable Diffusion, is designed to balance between a conditional input (e.g., a prompt) and an unconditional (neutral) generation process. The general formulations used are as follows:

- **CFG Paper Formulation:**

$$
\tilde{\epsilon_\theta}(z_t, t, c) = (1 + \omega) \epsilon_\theta(z_t, t, c) - \omega \epsilon_\theta(z_t, t, \emptyset)
$$

- **Diffusers Implementation:**

$$
\tilde{\epsilon_\theta}(z_t, t, c) = \epsilon_\theta(z_t, t, \emptyset) + \lambda (\epsilon_\theta(z_t, t, c) - \epsilon_\theta(z_t, t, \emptyset))
$$

  
Where:
- $z_t$ is the latent variable at timestep $t$,
- $c$ is the conditional input (e.g., prompt),
- $\emptyset$ represents the unconditional or neutral input,
- $\epsilon_\theta$ denotes the noise prediction model,
- $\omega$ is the guidance weight in the original CFG paper,
- $\lambda$ is the scaling factor in the diffusers implementation.

The relationship between the two formulations can be shown as $\lambda = 1 + \omega$, indicating they are functionally equivalent under this transformation. The difference is primarily in scaling: setting $\omega = 0$ in the CFG paper corresponds to $\lambda = 1$ in the diffusers' interpretation, meaning no guidance is applied.

### Current Negative Prompt Formulation in Stable Diffusion

Stable Diffusion incorporates a negative prompt mechanism, allowing users to specify undesired features in the generated images. The current implementation is:

$$
\tilde{\epsilon_\theta}(z_t, t, c_P) = \epsilon_\theta(z_t, t, c_N) + \lambda(\epsilon_\theta(z_t, t, c_P) - \epsilon_\theta(z_t, t, c_N))
$$

Where:
- $c_P$ is the positive prompt,
- $c_N$ is the negative prompt.

### Proposed Reformulation

To better align with the diffusers' CFG interpretation and enhance image generation control, we propose the following reformulation:

$$\tilde{\epsilon_\theta}(z_t, t, c) = \epsilon_\theta(z_t, t, \emptyset) + \lambda_P (\epsilon_\theta(z_t, t, c_P) - \epsilon_\theta(z_t, t, \emptyset)) + \lambda_N (\epsilon_\theta(z_t, t, \emptyset) - \epsilon_\theta(z_t, t, c_N))$$

In this formulation:
- $\lambda_P$ and $\lambda_N$ are the scaling factors for the positive and negative prompts, respectively.
- The formulation intuitively adds the difference between the positive prompt and unconditional noise while subtracting the difference between the negative prompt and unconditional noise.

This approach provides a more balanced and intuitive method to integrate both positive and negative prompts, potentially leading to higher quality and more controlled image generation.

## Implementation 
### Installation 

I tested with Python=3.10, torch=2.2.2, CUDA 12.2. 
```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install diffusers[torch] transformers matplotlib 
```

### Quick start

The whole code consists of modified DDIM scheduler (to integrate CFG++), pipeline, and notebook file. 
If you want to quickly start, just run 'run.ipynb'.


## Materials and Further Reading

For more information and a deeper understanding of the mechanisms discussed, the following resources are recommended:

- [Understanding the Impact of Negative Prompts: When and How Do They Take Effect?](https://minimaxir.com/2022/11/stable-diffusion-negative-prompt/)
- [Research Paper on CFG in Diffusion Models](https://arxiv.org/pdf/2406.02965)
- [CFG++: Manifold-constrained Classifier Free Guidance for Diffusion Models](https://arxiv.org/abs/2406.08070v1)

## Conclusion

This proposed reformulation of the negative prompt in Stable Diffusion, inspired by the diffusers' CFG interpretation, aims to provide improved control and image quality. By adjusting the relationship between the positive, negative, and unconditional inputs, this method offers a more nuanced and effective way to steer the generation process.

