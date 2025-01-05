import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

#Configuring our streamlit webpage
st.set_page_config(
    page_title="Graduation Admission Predictor",
    layout="centered"
)

def load_model():
    try:
        model = pickle.load(open('admissions_model.pkl','rb'))
        return model
    except Exception as e:
        st.error(f"Error Loading the model:{e}")
        return None
    
def predict_admission(model,features):
    try:
        prediction = model.predict([features])[0]
        return prediction
    except Exception as e:
        st.error(f"Error Making the prediction:{e}")
        return None

def main():
    st.title("Graduate Admission Chance Predictor")
    st.write("Enter your academic details to predict admission chances")

    with st.form("prediction_form"):
        gre = st.slider("GRE Score",260,340,300)
        toefl = st.slider("TOEFL Score",0,120,100)
        university_rating=st.slider("University Rating",1,5,3)
        sop = st.slider("Statement of Purpose",1.0,5.0,3.0,0.5)
        lor = st.slider("Letter of Recommendation",1.0,5.0,3.0,0.5)
        cgpa = st.slider("CGPA",6.0,10.0,8.0,0.1)
        research = st.radio("Research Experience",["Yes","No"])

        submitted = st.form_submit_button("Prdict Admission Chance")
        
        if submitted:
            model = load_model()

            if model:
                research_int = 1 if research == "Yes" else 0
                features = [gre,toefl,university_rating,sop,lor,cgpa,research_int]
                
                prediction = predict_admission(model,features)
                
                if prediction is not None:
                    st.success(f"Your prediction Chance of admission is {prediction*100:.2f}%")
                    
                    if prediction >= 0.8:
                        st.write("Excellent profile! Strong chances of admission")
                    elif prediction >=0.6:
                        st.write("Good Profile! Decent chances of Admission")
                    else:
                        st.write("Kindly strengthen your profile or applying to more universities")

                    st.info("Note: GRE,TOEFL and CGPA have the strongest impact on admission chances")

    st.markdown("How to improve your chances of admission:")
    st.write("""
    -Point number 1
    -Point number 2
             """)
    
if __name__ == "__main__":
    main()
    
    