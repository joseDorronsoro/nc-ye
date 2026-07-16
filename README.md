# Neural Collapse on Imbalanced MNIST/Fashion-MNIST

Framework for studying:

- Neural Collapse
- Class imbalance
- One-Hot Encoding (OHE)
- Ye target encoding
- Majority/minority performance
- Geometry of hidden representations

using a ResNet18 backbone trained on MNIST-like datasets.

---

# Motivation

This code originated from Neural Collapse experiments inspired by:

- Papyan, Han and Donoho (PNAS, 2020)
- Han, Papyan and Donoho (ICLR, 2022)

and was later extended to investigate:

- imbalanced classification
- alternative target encodings
- hidden-layer geometry
- majority/minority generalization

---

# Repository structure

```text
torch_resnet_bb.py        Main training script

fisher_functions.py       Neural Collapse metrics

results/
    ...
```

---

# Dataset format

Input datasets are stored as joblib Bunch objects.

Required fields:

```python
{
    "data":         ndarray,
    "target":       ndarray,

    "data_test":    ndarray,
    "target_test":  ndarray
}
```

---

# Imbalanced sampling

Training subsets are generated automatically.

The classes

```python
[0, 1, 2, 3, 4]
```

are considered majority classes.

The remaining classes

```python
[5, 6, 7, 8, 9]
```

are subsampled according to:

```bash
-fr
```

Example:

```bash
-fr 0.10
```

creates approximately a 10:1 imbalance ratio.

---

# Installation

Recommended environment:

```bash
python >= 3.10
```

Required packages:

```bash
pip install \
    torch \
    torchvision \
    numpy \
    scipy \
    scikit-learn \
    tqdm \
    joblib
```

---

# Command line usage

Example:

```bash
python torch_resnet_bb.py \
    -bf bunch_mnist.joblib \
    -e 350 \
    -bs 128 \
    -fr 0.5 \
    -en ohe \
    -l mse
```

Cluster example:

```bash
sbatch \
    -A gaa_serv \
    -p gaa \
    --exclude=casarrubuelos \
    --mail-type=ALL \
    --mail-user=<mail> \
    torch_resnet_bb.py \
    -bf bunch_mnist.joblib \
    -e 350 \
    -bs 128 \
    -fr 0.5 \
    -en ohe \
    -l mse \
    -rp 10
```

---

# Parameters

## Dataset

```bash
-bf
--bunch_file
```

Dataset filename.

Example:

```bash
-bf bunch_mnist.joblib
```

---

## Training

```bash
-bs
--batch_size
```

Mini-batch size.

Default:

```bash
128
```

---

```bash
-e
--epochs
```

Training epochs.

Example:

```bash
-e 350
```

---

```bash
-rp
--reps
```

Number of independent repetitions.

Default:

```bash
1
```

---

## Hardware

```bash
-g
--gpu
```

GPU identifier.

Example:

```bash
-g 0
```

---

## Target encoding

```bash
-en
--encoding
```

Allowed values:

```bash
ohe
ye
```

Example:

```bash
-en ye
```

---

## Loss

```bash
-l
--loss
```

Allowed values:

```bash
mse
ce
```

Examples:

```bash
-l mse
```

or

```bash
-l ce
```

---

## Optimizer

```bash
-ops
--optimizer_string
```

Allowed values:

```bash
sgd
adam
adamw
rmsp
```

---

## Imbalance fraction

```bash
-fr
--frac
```

Minority class fraction.

Example:

```bash
-fr 0.10
```

---

## Learning-rate scaling

```bash
-lrf
--lrate_factor
```

Examples:

```bash
-lrf 1.0
-lrf 2.0
```

---

## Saving results

```bash
-sv
```

Enable saving of experiment results.

Example:

```bash
-sv
```

Without this flag no results are stored.

---

# Encodings

## One-Hot Encoding (OHE)

Standard class encoding:

```text
0 -> [1,0,0,...]
1 -> [0,1,0,...]
...
```

---

## Ye Encoding

Targets depend on class probabilities.

Given

```math
p=(p_1,\ldots,p_C)
```

the target matrix is

```math
Y=
diag\left(
\frac{1}{\sqrt p}
\right)
-
\sqrt p \, 1^T
```

This produces probability-aware targets that adapt naturally to class imbalance.

---

# Model

Backbone:

```python
ResNet18
```

Modifications:

```python
conv1:
    kernel_size=3
    stride=1

maxpool:
    Identity()
```

which follows the common adaptation used for MNIST-size images.

---

# Outputs

The function:

```python
main(...)
```

returns:

```python
train_results
test_results
mse_history
```

---

## train_results

Structure:

```python
(
    targets,
    model_outputs,
    last_hidden_layer,
    classifier_weights
)
```

---

## test_results

Structure:

```python
(
    targets,
    model_outputs,
    last_hidden_layer,
    classifier_weights
)
```

---

# Neural Collapse analysis

After training, the script automatically performs:

```python
nc_analysis(...)
```

for:

- training data
- test data

The analysis operates on:

```python
last_hidden_layer
classifier_weights
```

returned by the network.

---

# Saved files

When:

```bash
-sv
```

is specified, the following files are written:

```text
*_train_results.joblib
*_test_results.joblib
*_l_mse.joblib
*_config.joblib
```

---

## config.joblib

Contains the complete experiment configuration.

Example:

```python
{
    "bunch_file": ...,
    "epochs": ...,
    "batch_size": ...,
    "encoding": ...,
    "frac": ...,
    ...
}
```

This allows exact reproduction of any stored run.

---

# Typical workflow

```text
Dataset
   ↓
Imbalanced subsampling
   ↓
Target encoding
   ↓
ResNet training
   ↓
Feature extraction
   ↓
Neural Collapse analysis
   ↓
Result saving
```

---

# Reproducibility checklist

For every experiment save:

- train_results.joblib
- test_results.joblib
- l_mse.joblib
- config.joblib

This guarantees that all numerical results can be reconstructed later.

---

# References

Papyan, V., Han, X. Y., Donoho, D. L.

Prevalence of Neural Collapse During the Terminal Phase of Deep Learning Training.

PNAS, 2020.

---

Han, X. Y., Papyan, V., Donoho, D. L.

Neural Collapse Under MSE Loss:
Proximity to and Dynamics on the Central Path.

ICLR, 2022.