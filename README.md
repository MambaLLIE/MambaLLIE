<p align="center">
    <img src="img/logo.jpg" width="300">
</p>

# MambaLLIE: Implicit Retinex-Aware Low Light Enhancement with Global-then-Local State Space

<!-- [Paper](https://arxiv.org/pdf/2405.16105v1) |  -->
[Project Page](https://mamballie.github.io/anon/)

<hr />


> **Abstract:** *Recent advances in low light image enhancement have been dominated by Retinex-based learning framework, leveraging convolutional neural networks (CNNs) and Transformers. However, the vanilla Retinex theory primarily addresses global illumination degradation and neglects local issues such as noise and blur in dark conditions. Moreover, CNNs and Transformers struggle to capture global degradation due to their limited receptive fields. While state space models (SSMs) have shown promise in the long-sequence modeling, they face challenges in combining local invariants and global context in visual data. In this paper, we introduce MambaLLIE, an implicit Retinex-aware low light enhancer featuring a global-then-local state space design. We first propose a Local-Enhanced State Space Module (LESSM) that incorporates an augmented local bias within a 2D selective scan mechanism, enhancing the original SSMs by preserving local 2D dependency. Additionally, an Implicit Retinex-aware Selective Kernel module (IRSK) dynamically selects features using spatially-varying operations, adapting to varying inputs through an adaptive kernel selection process. Our Global-then-Local State Space Block (GLSSB) integrates LESSM and IRSK with layer normalization (LN) as its core. This design enables MambaLLIE to achieve comprehensive global long-range modeling and flexible local feature aggregation. Extensive experiments demonstrate that MambaLLIE significantly outperforms state-of-the-art CNN and Transformer-based methods. * 
>

<p align="center">
  <img width="800" src="img/pipeline.png">
</p>

---

## Installation

Please see [Conda.sh] for the installation of dependencies required to run MambaLLIE.

## Training and Evaluation

Our models and weights are coming soon. It might take me some time to sort it.

```shell
# activate the environment
conda activate MambaLLIE

# Train LOL-v2-real
python3 basicsr/train.py --opt options/train/LOLv2_real.yml

# Test LOL-v2-real
python3 basicsr/test.py --opt options/test/LOLv2_real.yml

```

## Acknowledgement

This code is based on [BasicSR](https://github.com/XPixelGroup/BasicSR), [MambaIR](https://github.com/csguoh/MambaIR) ,and [Retinexformer](https://github.com/caiyuanhao1998/Retinexformer), [SNR-Net](https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance) [LSKNet](https://github.com/zcablii/LSKNet). Thanks for their awesome work. -->
