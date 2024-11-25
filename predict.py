import streamlit as st
import pickle
import pandas as pd
import xgboost as xgb
import pdb
import shap
import matplotlib.pyplot as plt

input_data = {
    "Sex": "Male",
    "Age": 1,
    "ECOG PS Score": 1,
    "Tumor Size (cm)": 0.1,
    "EGFR": "No",
    "ALK": "No",
    "STAS": "No",
    "VPI": "No",
    "PMI": "No",
    "LVI": "No",
    "PNI": "No",
    "Tumor High-Grade Components >20%": "No",
    "Smoked in the Last 2 Years": "No",
    "CEA Value Detected": 0.01,
    "Post-Operative Chemotherapy": "No",
}

def calculate(v):
    with open(r"xgboost_model.pkl", 'rb') as f:
        model = pickle.load(f)

    a = []
    if v[0]=='Male':
        a.append(1)
    else:
        a.append(0)

    if v[1]>70:
        a.append(1)
    else:
        a.append(0)

    if v[2]>=90:
        a.append(1)
    else:
        a.append(0)

    if v[3]>=2.0:
        a.append(1)
    else:
        a.append(0)

    if v[4]=='Yes':
        a.append(1)
    else:
        a.append(0)

    if v[5]=='Yes':
        a.append(1)
    else:
        a.append(0)

    if v[6]=='Yes':
        a.append(1)
    else:
        a.append(0)

    if v[7]=='Yes':
        a.append(1)
    else:
        a.append(0)

    if v[8]=='Yes':
        a.append(1)
    else:
        a.append(0)

    if v[9]=='Yes':
        a.append(1)
    else:
        a.append(0)

    if v[10]=='Yes':
        a.append(1)
    else:
        a.append(0)

    if v[11]=='Yes':
        a.append(1)
    else:
        a.append(0)

    if v[12]=='Yes':
        a.append(1)
    else:
        a.append(0)

    if v[13]>4.7:
        a.append(1)
    else:
        a.append(0)

    if v[14]=='Yes':
        a.append(1)
    else:
        a.append(0)

    b=[]
    b.append(a)
    traindic={'f1':[1],'f2':[0],'f3':[1],'f4':[0],'f5':[1],'f6':[0],'f7':[1],'f8':[0],'f9':[1],'f10':[0],'f11':[1],'f12':[0],'f13':[1],'f14':[0],'f15':[1]}
    df = pd.DataFrame(traindic)
    da=pd.DataFrame(b,columns=df.columns)
    da=xgb.DMatrix(da)

    y_pre = model.predict(da)
    return y_pre[0],model,da,b
 

def main():
    st.title("Recurrence prediction")
    st.write("A model to predict the recurrence rate of Stage I non-small cell lung adenocarcinoma")

    with st.form("input_form"):
        for key in input_data.keys():
            if isinstance(input_data[key], int):
                input_data[key] = st.number_input(key, min_value = 1,  value=input_data[key], step=1)
            elif isinstance(input_data[key], float):
                if key == "Tumor Size (cm)":
                    input_data[key] = st.number_input(key,min_value = 0.1, value=input_data[key], step=0.1, format="%.1f") 
                else:
                    input_data[key] = st.number_input(key,min_value=0.01, value=input_data[key], step=0.01, format="%.2f")
            else:
                if "Detected" in key:
                    options = ["No", "Yes"]
                elif "Is" in key:
                    options = ["No", "Yes"]
                elif key == "Sex":
                    options = ["Male", "Female"]
                else:
                    options = ["No", "Yes"]
                input_data[key] = st.selectbox(key, options)

        submitted = st.form_submit_button("Submit")
        if submitted:
            v = list(input_data.values())
            result, model, da, b = calculate(v)

            shap.initjs()
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(da)
            
            shap.force_plot(explainer.expected_value, shap_values, b[0],matplotlib=True)
        
            st.success(f"Predicted probability of recurrence: {result:.2f}")
            st.pyplot(plt)

if __name__ == "__main__":
    main()
