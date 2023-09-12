# Density-Aware Calibration [ICML 2023] (official)
Official implementation of **Density-Aware Calibration (DAC)** presented in, 
["Beyond In-Domain Scenarios: Robust Density-Aware Calibration."](https://arxiv.org/abs/2302.05118), **ICML 2023**. 

### Approach
<img src="https://github.com/futakw/DensityAwareCalibration/blob/master/teaser_fig.png" width=60% height=60%>
  
DAC can be combined with any existing post-hoc calibration method $h$, leading to robust and reliable uncertainty estimates, especially under domain-shift scenarios.

DAC leverages information from feature vectors $z_1,...,z_L$ across the entire classifier $f$. DAC is based on KNN, where predictive uncertainty is expected to be high for test samples lying in low-density regions of the empirical training distribution and vice versa.

### Reproducible results from this repo.
In this repo, you can reproduce the following results for calibration of CIFAR10, ResNet18:
We combine DAC with [ETS](https://github.com/zhang64-llnl/Mix-n-Match-Calibration) and [SPL](https://github.com/kartikgupta-at-anu/spline-calibration).
- ETS vs. ETS+DAC (ours).
```
(ECE, the lower the better)
+----+------------------+----------+---------------+-------------+
|    | test data        |   Uncal. |   ETS w/o DAC |   ETS + DAC |
|----+------------------+----------+---------------+-------------|
|  0 | natural_1        | 0.105783 |     0.0140503 |  0.00914038 |
|  1 | gaussian_noise_3 | 0.19937  |     0.049552  |  0.0417357  |
|  2 | gaussian_noise_5 | 0.256887 |     0.0895087 |  0.0737431  |
+----+------------------+----------+---------------+-------------+
```
- SPL vs. SPL+DAC (ours).
```
(ECE, the lower the better)
+----+------------------+----------+---------------+-------------+
|    | test data        |   Uncal. |   SPL w/o DAC |   SPL + DAC |
|----+------------------+----------+---------------+-------------|
|  0 | natural_1        | 0.105783 |     0.0212573 |   0.0110498 |
|  1 | gaussian_noise_3 | 0.19937  |     0.0626797 |   0.0462735 |
|  2 | gaussian_noise_5 | 0.256887 |     0.100396  |   0.0818538 |
+----+------------------+----------+---------------+-------------+
```

### Citation
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
- 1.1. Download example features from google drive (187.4 MB)
```
pip3 install gdown
gdown https://drive.google.com/uc?id=1L2rY0FM32D_xd5ERn7h4VVdPPlwoNQos
unzip outputs.zip
rm outputs.zip
```
- 2.2. Run demo
```
source venv/bin/activate
bash scripts/quick_comparison_DAC_vs_ETS.sh
```
This compares ETS vs ETS+DAC for CIFAR10-ResNet18.
(SPL vs SPL+DAC, by "bash scripts/quick_comparison_DAC_vs_SPL.sh")

### 2. Whole pipeline: Extract features + run DAC
- 2.1. Download CIFAR-10-C dataset from https://zenodo.org/record/2535967
```
mkdir data
wget -O data/CIFAR-10-C.tar https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1
tar -xvf data/CIFAR-10-C.tar -C data
```
- 2.2. Extract features + run DAC
```
source venv/bin/activate
bash scripts/whole_pipeline.sh
```
("bash scripts/whole_pipeline_all.sh" to extract features for all corruption data)

## Calibrate your own classifier
### 1. Add your model to utils/get_models.py
### 2. Check the names of layers
You can modify and run "python3 utils/get_models.py", or simply run the following.
```
model = get_model(**args)
print(model)

# check available layers
for name, module in model.named_modules():
    print("- ", name)
```
### 3. Select layers to use in DAC.
Instead of using only logits, as existing post-hoc methods do, DAC uses features from intermediate layers of the classifier that have calibration-related information.
```
For neural networks that have block structures (most neural networks do), we recommend to pick,
- (1) the very last layer of each block
- (2) the immediate layer before the first block
- (3) the layer just before the fully-connected layers at the end of neural networks
- (4) the logits layer 
(Please see Section C in the Appendix of our paper for detailed explanation). Although it is also possible to use all layers, we recommend to select the layers with the above strategy for much faster optimization and calibration.
```

Once you selected the layers, please put the name list of layers in "constants/layer_names.py", so that "get_layers_name()" returns the layer names. For example, 
```
if "resnet" in model_name:
    return_nodes = [
        "maxpool",
        "layer1",
        "layer2",
        "layer3",
        "layer4",
    ]
```

### 4. Modify "scripts/whole_pipeline.sh" and run.
You should specify the model_name and model_path. 
If necessary, modify "utils/dataset.py" and specify the dataset name.



## Structure of this repository
- density_aware_calib.py: Our method is here.
- utils: Some codes imported and modified from other repos.
    - Evaluation codes, ETS: from https://github.com/zhang64-llnl/Mix-n-Match-Calibration
    - SPL: from https://github.com/kartikgupta-at-anu/spline-calibration


## Note: How to store features/logits/labels to try our repository.
features/ood_score, outputs (logits), labels from a classifier will be saved in the following structure, by running _extract_feature_and_knn_score.py_:
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
