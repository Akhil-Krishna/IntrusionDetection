import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import joblib
import io


arff_file_path = "KDDTrain+.arff"
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

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and save SVM model
svm_model = SVC(kernel='rbf', probability=True)
svm_model.fit(X_train, y_train)
joblib.dump(svm_model, "svm_model.pkl")

# Train and save Decision Tree model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
joblib.dump(dt_model, "dt_model.pkl")

print("Models trained and saved successfully!")
