import joblib
import json
import numpy as np

print("Loading model...")
model = joblib.load('ml_models/rf_model.pkl')
print("Model loaded.")

forest_data = []
for tree in model.estimators_:
    tree_ = tree.tree_
    forest_data.append({
        'children_left': tree_.children_left.tolist(),
        'children_right': tree_.children_right.tolist(),
        'feature': tree_.feature.tolist(),
        'threshold': tree_.threshold.tolist(),
        # We only need the values, flattened
        'value': [v[0].tolist() for v in tree_.value] 
    })

# Convert numpy int64/float64 to python native types if any remain
def convert(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32): return int(o)
    if isinstance(o, np.float64) or isinstance(o, np.float32): return float(o)
    raise TypeError

output = {
    'classes': model.classes_.tolist(),
    'trees': forest_data
}

with open('ml_models/rf_model_light.json', 'w') as f:
    json.dump(output, f, default=convert)
print("Saved JSON.")
