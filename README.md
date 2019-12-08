# Lethean Attack: An Online Data Poisoning Technique

Heavily based on the source code by `Y. Sun, X. Wang, Z. Liu, J. Miller, A. A. Efros, and M. Hardt. Test-time training for out-of-distribution generalization, 2019`

See [paper](link to paper) for more details.

## Prerequisites

- Python 3.5+
- PyTorch
- torchvision

## Fetch Data

CIFAR-10 is fetched automatically using torchvision. To download CIFAR-10-C (used by `test.py`):
```
mkdir -p data && cd data
wget https://zenodo.org/record/2535967/files/CIFAR-10-C.tar
tar xvf CIFAR-10-C.tar
```

## Train a Test-Time Training Model

```
python main.py --batch_size 128 --outf [MODEL_DIRECTORY] --epochs 150
```

## Perform a Lethean Attack

```
python adversarial_lethean.py --resume [MODEL_DIRECTORY] --epochs 5000
```

## Perform an FGSM Attack

```
python adversarial_fgsm.py --resume [MODEL_DIRECTORY] --epochs 5000
```

## Test using CIFAR-10-C data

```
python adversarial_fgsm.py --resume [MODEL_DIRECTORY] --epochs 5000 --corruption [CORRUPTION] --level [1/2/3/4/5]
```

Possible corruption types:
- brightness
- defocus_blur
- fog
- gaussian_blur
- glass_blur
- jpeg_compression
- motion_blur
- saturate
- snow
- speckle_noise
- contrast
- elastic_transform
- frost
- gaussian_noise
- impulse_noise
- pixelate
- shot_noise
- spatter
- zoom_blur
