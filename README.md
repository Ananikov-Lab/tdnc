# Totally Defined Nanocatalysis

The repository contains the code for the article **"Towards Totally Defined Nanocatalysis: Deep Learning Reveals the Extraordinary Activity of Single Pd/C Particles"** by *Dmitry B. Eremin, Alexey S. Galushko, Daniil A. Boiko, Evgeniy O. Pentsak,  Igor V. Chistyakov, and Valentine P. Ananikov*

- **tdnc/** — a Python module, which contains useful functions and classes for training segmentation models for SEM image analysis
- **configs/** — task-specific configuration files, allowing to reproduce the models
- **train.py** — the segmentation model training script
- **inference.py** – the code to perform inference of the models
- **configs/augmentations/** – augmentations, specific for each material

All models can be retrained with the following command
```bash
for f in configs/*.yaml; do python train.py --config $f; done
```

The training was performed on a single NVIDIA 1080 TI.

[<img src="https://pbs.twimg.com/profile_images/678945177592049664/NwCxumCE_200x200.png" alt="Ananikov Lab" width="100"/>](ananikovlab.ru)