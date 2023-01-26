# Set up
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Run Demo
- Download example features from google drive (4GB)
```
pip3 install gdown
gdown https://drive.google.com/uc?id=1aAMlTQUqjiBnUT814_nOT-z2sFwDi7l9
unzip outputs.zip
```
- run demo
```
bash demo.sh
```
This compares ETS vs ETS+DAC for CIFAR10-ResNet18.
(Also SPL vs SPL+DAC by changing arguments)

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