import joblib
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import warnings
warnings.filterwarnings('ignore')

print("Loading model...")
model = joblib.load('ml_models/rf_model.pkl')
symptoms = joblib.load('ml_models/symptoms_list.pkl')

print("Converting model to ONNX...")
# Input type is a float tensor of shape [None, len(symptoms)]
initial_type = [('float_input', FloatTensorType([None, len(symptoms)]))]
onx = convert_sklearn(model, initial_types=initial_type, target_opset=12)

print("Saving ONNX model...")
with open("ml_models/rf_model.onnx", "wb") as f:
    f.write(onx.SerializeToString())

print("Saved ml_models/rf_model.onnx!")
