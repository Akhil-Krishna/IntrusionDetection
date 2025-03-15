from sklearn.metrics import accuracy_score
import streamlit as st
import pandas as pd
import joblib

# Load pre-trained models (Ensure these are in the same directory)
svm_model = joblib.load("svm_model.pkl")  # Load SVM model
dt_model = joblib.load("dt_model.pkl")  # Load Decision Tree model
dataList=[
    [2,0,1,0,12983,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0.00,0.00,0.00,0.00,1.00,0.00,0.00,134,86,0.61,0.04,0.61,0.02,0.00,0.00,0.00,0.00],#0
    [0,0,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,229,10,0.00,0.00,1.00,1.00,0.04,0.06,0.00,255,10,0.04,0.06,0.00,0.00,0.00,0.00,1.00,1.00],#1
]

raw_list=[
    [0,"tcp","private","REJ",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,229,10,0.00,0.00,1.00,1.00,0.04,0.06,0.00,255,10,0.04,0.06,0.00,0.00,0.00,0.00,1.00,1.00,"anomaly"],
    [2,"tcp","ftp_data","SF",12983,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0.00,0.00,0.00,0.00,1.00,0.00,0.00,134,86,0.61,0.04,0.61,0.02,0.00,0.00,0.00,0.00,"normal"]
]

for i in dataList:
    prediction = dt_model.predict([i])
    print(prediction)