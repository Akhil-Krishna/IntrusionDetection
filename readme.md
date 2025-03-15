# README

## 1. Data Preprocessing
I used normal methods and didn't change column names to predict the Attack type. 
I only used some encoding, for example, converting `tcp` to `0`, and setting `anomaly = 1` and `normal = 0`.

## 2. Training
I trained the SVM and Decision Tree models to predict whether an anomaly exists or not.

**But if you have created SVM and DT models to predict attack type, simply save the model as:**
```python
import joblib
joblib.dump(svm_model, "svm_model.pkl")
joblib.dump(dt_model, "dt_model.pkl")
```

## 3. Website (Can Customize UI)
Simply use:
```sh
streamlit run app.py
```

## Keep in Mind
If another prediction model is needed, simply use the respective `model.pkl` in `app.py`.