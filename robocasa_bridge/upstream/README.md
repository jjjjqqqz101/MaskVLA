# Third-party code in the bridge

- The EGL compatibility patch modifies `robosuite/utils/binding_utils.py` from robosuite commit `5ce6643f3092639d08f7b0f90ed1c6a84f50552c`. It allows EGL and CUDA device numbering to differ. The patch includes context from robosuite. [robosuite MIT license](ROBOSUITE-LICENSE.txt)
- The π0.5 pilot's flow objective and optional padding ablation adapt the OpenPI flow-matching and Euler sampling code from Physical Intelligence. The changes restrict the loss to the seven active robot controls and optionally constrain unused sampling dimensions. [OpenPI](https://github.com/Physical-Intelligence/openpi) · [Apache License 2.0](OPENPI-LICENSE.txt)
- The ACT pilot extends the ManiSkill ACT baseline's DETRVAE forward pass with a padding-aware encoder and reconstruction loss. The underlying DETR implementation carries the Facebook, Inc. and affiliates copyright notice. [ManiSkill](https://github.com/haosulab/ManiSkill) · [Apache License 2.0](OPENPI-LICENSE.txt)
