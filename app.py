import streamlit as st
import pandas as pd
import joblib

# Load pre-trained models (Ensure these are in the same directory)
svm_model = joblib.load("svm_model.pkl")  # Load SVM model
dt_model = joblib.load("dt_model.pkl")  # Load Decision Tree model

# Streamlit UI Layout
st.markdown(
    """
    <style>
        .stButton>button {background-color: #004466; color: white; border-radius: 5px;}
        .stTextInput, .stTextArea, .stFileUploader, .stSelectbox {width: 100%;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("AI IDS Attack Prediction")

# Dropdown 
model_choice = st.selectbox("Select Model for Prediction:", ["SVM", "Decision Tree"])

# Upload CSV Section
st.subheader("Upload CSV File for Prediction")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
if st.button("Upload File"):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        """
        Here add any logic for manipuating CSV file (Ippo tcp to 0 , icmp to 1  encoding) , 
        
        also add arff to csv conversion if needed
        
        Evide CSV file character values must encode to Numeric ones , 
        Else : value kodukunnathe agane aarikanam
        
        You must provide like [
    [2,0,1,0,12983,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0.00,0.00,0.00,0.00,1.00,0.00,0.00,134,86,0.61,0.04,0.61,0.02,0.00,0.00,0.00,0.00,#0]
    ]

    Instead of 
      [2,"tcp","ftp_data","SF",12983,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0.00,0.00,0.00,0.00,1.00,0.00,0.00,134,86,0.61,0.04,0.61,0.02,0.00,0.00,0.00,0.00,"normal"]
      
        
        """
        st.write("File Uploaded Successfully!")
        st.write(df.head())  # Show the first few rows
    else:
        st.error("Please upload a CSV file.")

# Enter Payload for Prediction
st.subheader("Or Enter Payload for Prediction")
st.write("Must give encoded Values")
st.write('[0,"tcp","private","REJ",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,229,10,0.00,0.00,1.00,1.00,0.04,0.06,0.00,255,10,0.04,0.06,0.00,0.00,0.00,0.00,1.00,1.00]')
st.write("must be converted to ")
st.write("[0,0,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,229,10,0.00,0.00,1.00,1.00,0.04,0.06,0.00,255,10,0.04,0.06,0.00,0.00,0.00,0.00,1.00,1.00] ")
payload = st.text_area("Enter comma-separated values")

if st.button("Predict"):
    if payload:
        try:
        
            input_data = pd.DataFrame([list(map(float, payload.split(",")))])
            
            model = svm_model if model_choice == "SVM" else dt_model
            
        
            prediction = model.predict(input_data)
            
            
            st.subheader("Prediction Results")
            attack=['Normal',"Anomaly"]
            st.write(f"**Attack : {attack[prediction[0]]}**")
        except Exception as e:
            st.error(f" Error: {e}")
    else:
        st.error("Please enter values for prediction.")
