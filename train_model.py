import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

print("🚀 Starting Logistic Regression Model Training...")

os.makedirs('ml_models', exist_ok=True)

print("Loading your full dataset...")
df = pd.read_csv("Final_Augmented_dataset_Diseases_and_Symptoms.csv")

X = df.drop("diseases", axis=1)
y = df["diseases"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training the memory-efficient model (Please wait 1-2 minutes)...")

model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 FINAL MODEL ACCURACY: {accuracy * 100:.2f}%")

print("Saving the file safely...")

joblib.dump(model, 'ml_models/rf_model.pkl', compress=9)
joblib.dump(list(X.columns), 'ml_models/symptoms_list.pkl') 

print("\n✅ Model saved securely in 'ml_models/'!")