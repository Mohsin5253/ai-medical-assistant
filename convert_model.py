import joblib
import m2cgen as m2c
print("Loading model...")
model = joblib.load('ml_models/rf_model.pkl')
print("Converting to Python code...")
code = m2c.export_to_python(model)
with open('ml_models/rf_model_code.py', 'w') as f:
    f.write(code)
print("Done!")
