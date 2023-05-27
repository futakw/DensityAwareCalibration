# Density Aware Calibration (official)
This is an official implementation Density-Aware Calibration (DAC).
This method is presented in the paper "Beyond In-Domain Scenarios: Robust Density-Aware Calibration.", ICML 2023. 
(arXiv link: https://arxiv.org/abs/2302.05118)

If you find this repository useful, please cite our paper.
```
@article{tomani_waseda2023beyond,
  title={Beyond In-Domain Scenarios: Robust Density-Aware Calibration},
  author={Tomani, Christian and Waseda, Futa and Shen, Yuesong and Cremers, Daniel},
  journal={ICML 2023},
  year={2023}
}
```

# Set up
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Note:
By default faiss-gpu is installed, which requires GPU resource. 
If you don't use GPU, please uninstall faiss-gpu and install faiss-cpu.

# Run Demo: CIFAR10, ResNet18
## Quick Start: Download features + run DAC
1. Download example features from google drive (4GB)
```
pip3 install gdown
gdown https://drive.google.com/uc?id=1aAMlTQUqjiBnUT814_nOT-z2sFwDi7l9
unzip outputs.zip
```
2. run demo
```
source venv/bin/activate
bash demo.sh
```
This compares ETS vs ETS+DAC for CIFAR10-ResNet18.
(Also SPL vs SPL+DAC by changing arguments)

Expected results:
Calibration performance on test set: gaussian_noise_5
- ETS w/o DAC: test_ece_1: 0.08950865
- ETS + DAC: test_ece_1: 0.07374325

## Whole pipeline: Extract features + run DAC
1. Download CIFAR-10-C dataset from https://zenodo.org/record/2535967
```
cd data
wget -O CIFAR-10-C.tar https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1
tar -xvf CIFAR-10-C.tar
```
2. 


## Inference pipeline: 



# Structure
- density_aware_calib.py: Our method is here.
- utils: Some codes imported from others, and modified.
    - Evaluation codes, ETS: from https://github.com/zhang64-llnl/Mix-n-Match-Calibration
    - SPL: from https://github.com/kartikgupta-at-anu/spline-calibration

# Note:
To run experiments, features, outputs (logits), labels from the classifier should be stored as, for example,
```
Features: outputs/{dataset_name}/{classifier_name}/{data_type}/features/{layer_name}.pickle
Logits: outputs/{dataset_name}/{classifier_name}/{data_type}/outputs.pickle
Labels: outputs/{dataset_name}/{classifier_name}/{data_type}/labels.pickle
```