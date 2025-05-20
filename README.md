
# Parameteric DeepONet for modeling structural dynamics


## Installation
Deep operator networks (DeepONet) is based on the [DeepXDE](https://github.com/lululxvi/deepxde). Install DeepXDE by:
```bash
pip install deepxde
```


## Data
### Case1: The Duffing oscillator
Please refer to `./data/` folder.

### Case2: The wind turbine blade


## Train

### Forward modeling
Experiments of Case1 and Case2 are in the `./experiments/` folder.

For instance, the Case 1b in `case1b_forward.ipynb`.

For other training configurations, you can set the training setting via `.yaml` config file, and start training by the `run.sh`.
```bash
sh run.sh
```

### Inverse modeling
Try the example of Case 1b in `case1b_inverse.ipynb`.



### Acknowledgement
Our code is parially based on:
\
https://github.com/lululxvi/deepxde 
\
https://github.com/adler-j/learned_gradient_tomography
\
https://github.com/csiro-mlai/fno_inversion_ml4ps2021


This repository is the code and data for the manuscript "[Parameter estimation of structural dynamics with neural operators enabled surrogate modeling](https://arxiv.org/abs/2410.11712)".