
# Parameteric DeepONet for modeling structural dynamics
This repository is the code and data for the manuscript: [Parameter estimation of structural dynamics with neural operators enabled surrogate modeling](https://arxiv.org/abs/2410.11712).

## Installation
Deep operator networks (DeepONet) is based on the [DeepXDE](https://github.com/lululxvi/deepxde). Install DeepXDE by:
```bash
pip install deepxde
```
The implementations are based on Ubuntu 20.04, Python 3.8, and PyTorch-Cuda 11.6.



## Data

Please refer to `./data/` folder.


## Train the foward surrogate model
Set the training configurations via `.yaml` config file, and start training by the `run.sh`.
```bash
sh run.sh
```

## Evaluate
Experiments of Case1 and Case2 are in the `./experiments/` folder. 


### Case 1
SDOF Response prediction, see example in `case1b_forward.ipynb`.

Parameter estimation, see example inin `case1b_inverse.ipynb`.

### Case 2

MDOF response prediction, see example in `case2_forward_params_a.ipynb`.

Damage length estimation, see example in `case2_inverse_params_a.ipynb`.

Damage shape estimation, see example in `case2_inverse_params_b.ipynb`.

(Set the suitable parameterization code in `inverse_net.py` and `script.py`)

## Acknowledgement
Our code is partially based on:
\
https://github.com/lululxvi/deepxde 
\
https://github.com/adler-j/learned_gradient_tomography
\
https://github.com/csiro-mlai/fno_inversion_ml4ps2021


