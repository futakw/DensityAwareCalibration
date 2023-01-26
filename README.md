# Set up
```
pip install -r requirements.txt
```

# Run Demo
```
demo.sh
```
This compares ETS vs ETS+DAC.

# Structure
- density_aware_calib.py: our method is here.

# Note:
To run experiments, features/outputs/labels from the classifier should be stored as,
```
Features: outputs/{dataset_name}/{model_name}/{data_type}/features/{layer_name}.pickle
Logits: outputs/{dataset_name}/{model_name}/{data_type}/outputs.pickle
Labels: outputs/{dataset_name}/{model_name}/{data_type}/labels.pickle
```