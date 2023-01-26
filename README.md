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

# Structure
- density_aware_calib.py: our method is here.
- utils: some codes imported by others and modified, for evaluation, ETS, SPL

# Note:
To run experiments, features/outputs/labels from the classifier should be stored as,
```
Features: outputs/{dataset_name}/{model_name}/{data_type}/features/{layer_name}.pickle
Logits: outputs/{dataset_name}/{model_name}/{data_type}/outputs.pickle
Labels: outputs/{dataset_name}/{model_name}/{data_type}/labels.pickle
```