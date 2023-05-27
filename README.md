# Density-Aware Calibration, ICML 2023 (official from author)
Official implementation of Density-Aware Calibration (DAC), presented in, "Beyond In-Domain Scenarios: Robust Density-Aware Calibration.", ICML 2023. 
(arXiv: https://arxiv.org/abs/2302.05118)

### Approach
![fig1](figures/teaser_fig_1.png)
![fig2](figures/teaser_fig_2.png)

DAC boosts the calibration performance of the existing post-hoc calibration methods, especially in the domain-shift scenario.
DAC leverages information from feature vectors $z_1,...,z_L$ across the entire classifier $f$. DAC is based on KNN, where predictive uncertainty is expected to be high for test samples lying in low-density regions of the empirical training distribution and vice versa.


In this repo, you can reproduce the following results.

Calibration results for CIFAR10, ResNet18:
- ETS vs. ETS+DAC (ours).
```
(ECE, the lower the better)
+----+------------------+---------------+-------------+ 
|    | test data        |   ETS w/o DAC |   ETS + DAC |
|----+------------------+---------------+-------------|
|  0 | natural_1        |    0.0140503  |  0.00914035 |
|  1 | gaussian_noise_3 |    0.049552   |  0.0417358  |
|  2 | gaussian_noise_5 |    0.0895087  |  0.0737432  |
+----+------------------+---------------+-------------+
```
- SPL vs. SPL+DAC (ours).
```
(ECE, the lower the better)
+----+------------------+---------------+-------------+ 
|    | test data        |   SPL w/o DAC |   SPL + DAC |
|----+------------------+---------------+-------------|
|  0 | natural_1        |     0.0212603 |  0.0110497  |
|  1 | gaussian_noise_3 |     0.0626791 |  0.0462725  |
|  2 | gaussian_noise_5 |     0.100395  |  0.0818528  |
+----+------------------+---------------+-------------+
```

### Cite our paper
If you find this repository useful, please cite our paper:
```
@article{tomani_waseda2023beyond,
  title={Beyond In-Domain Scenarios: Robust Density-Aware Calibration},
  author={Tomani, Christian and Waseda, Futa and Shen, Yuesong and Cremers, Daniel},
  journal={ICML},
  year={2023}
}
```

# Usage
## Environment set up
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Note:
By default faiss-gpu is installed, which requires GPU resource. 
If you don't use GPU, please uninstall faiss-gpu and install faiss-cpu.

## Run Demo: CIFAR10, ResNet18
### 1. Quick Start: Download features + run DAC
1.1. Download example features from google drive (4GB)
```
pip3 install gdown
gdown https://drive.google.com/uc?id=1aAMlTQUqjiBnUT814_nOT-z2sFwDi7l9
unzip outputs.zip
```
2.2. run demo
```
source venv/bin/activate
bash scripts/quick_comparison_DAC_vs_ETS.sh
(bash scripts/quick_comparison_DAC_vs_SPL.sh)
```
This compares ETS vs ETS+DAC for CIFAR10-ResNet18.
(Also SPL vs SPL+DAC by changing arguments)

### 2. Whole pipeline: Extract features + run DAC
2.1. Download CIFAR-10-C dataset from https://zenodo.org/record/2535967
```
cd data
wget -O CIFAR-10-C.tar https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1
tar -xvf CIFAR-10-C.tar
```
2.2. Extract features + run DAC
```
source venv/bin/activate
bash scripts/whole_pipeline.sh
```

### 3. Inference pipeline: 



# Structure of this repository
- density_aware_calib.py: Our method is here.
- utils: Some codes imported from others, and modified.
    - Evaluation codes, ETS: from https://github.com/zhang64-llnl/Mix-n-Match-Calibration
    - SPL: from https://github.com/kartikgupta-at-anu/spline-calibration

# Note: structure of saved features/ood_score, outputs, and labels
extract_feature_and_knn_score.py:
    features/ood_score, outputs (logits), labels from the classifier will be saved in the following structure.
```
- outputs
    - {dataset_name}
        - {classifier_name}
            - train # used for KNN scoring.
                - features
                    - {layer_name_1}.pickle
                    - {layer_name_2}.pickle
                    - ...
                outputs.pickle # logits
                labels.pickl # labels
            - val # used for calibration optimization
                - ood_score # Optionally, "features" or "ood_score_arr"
                    - {layer_name_1}.pickle
                    - {layer_name_2}.pickle
                    - ...
                outputs.pickle # logits
                labels.pickl # labels
            - {test_data_name}
                - ood_score # Optionally, "features" or "ood_score_arr"
                    - {layer_name_1}.pickle
                    - {layer_name_2}.pickle
                    - ...
                outputs.pickle # logits
                labels.pickl # labels
```