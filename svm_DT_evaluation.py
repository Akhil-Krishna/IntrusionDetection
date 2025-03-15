from sklearn.metrics import accuracy_score
import streamlit as st
import pandas as pd
import joblib

# Load pre-trained models (Ensure these are in the same directory)
svm_model = joblib.load("svm_model.pkl")  # Load SVM model
dt_model = joblib.load("dt_model.pkl")  # Load Decision Tree model


import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder, StandardScaler
import io


# test data : i am converting to std format
arff_file_path = "KDDTest+.arff"
with open(arff_file_path, "r", encoding="utf-8") as f:
    arff_data = f.read()


# icmp issue here
arff_data_fixed = arff_data.replace(" 'icmp'", "icmp")


arff_file = io.StringIO(arff_data_fixed)
data = arff.loadarff(arff_file)
df = pd.DataFrame(data[0])


def decode_and_strip(value):
    if isinstance(value, bytes):
        return value.decode('utf-8').strip()
    return str(value).strip()

for col in df.select_dtypes([object]):
    df[col] = df[col].apply(decode_and_strip)


target_column = 'class'  
y = df[target_column]
X = df.drop(columns=[target_column])

# Encode categorical features
label_encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

y_le = LabelEncoder()
y = y_le.fit_transform(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)


y_prediction_dt = dt_model.predict(X)
y_prediction_svm = svm_model.predict(X)

# Evaluate accuracy
accuracy_svm = accuracy_score(y, y_prediction_svm)
accuracy_dt = accuracy_score(y, y_prediction_dt)
print(f"SVM Model Accuracy          : {accuracy_svm:.4f}")
print(f"Decision Tree Model Accuracy: {accuracy_dt:.4f}")


"""

SVM Model Accuracy          : 0.8177
Decision Tree Model Accuracy: 0.7421

"""