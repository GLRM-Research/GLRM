# GLRM: Global and Local Riemannian Representations for Motor Imagery EEG Decoding

GLRM is a dual-branch framework for motor imagery EEG decoding. It combines global Riemannian representations of whole-window spatial statistics with local covariance sequences modeled by a Mamba-based selective state-space encoder. The independently trained branches are combined through dual-branch fusion.

## Current release

This repository provides the **core model implementation only**:

- Covariance feature encoding and Riemannian representation modules.
- Global static Riemannian branch (GSR).
- Local dynamic Riemannian branch (LDRM).
- Dual-branch fusion and the GLRM model interface.

The current release does not include dataset processing pipelines, experimental splits, training or evaluation scripts, experiment configurations, datasets, or pretrained weights.

## Code release plan

**Following acceptance of the paper, we will progressively release additional code, including data preprocessing, training and evaluation pipelines, configurations, and usage instructions.**

## Model source

The model components are located in `glr_ssm/`. The public model interface is `GLRM`; `GLRSSM` is retained as a compatibility alias. `requirements.txt` lists the core dependencies and does not specify the experimental environment.

## License and dependencies

The repository code is released under the [MIT License](LICENSE). Mamba is used as an external dependency; its source code and license are available at [state-spaces/mamba](https://github.com/state-spaces/mamba). Third-party dependencies remain subject to their respective licenses.
