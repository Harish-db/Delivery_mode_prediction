#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as rfclassifier


# In[3]:


st.title('Delivery of a Pregnant women Prediction')

with open ('df_top8_df.pkl','rb') as file:
    df_data=pickle.load(file)
    
with open ('rfclassifier.pkl','rb') as file:
    rfclassifier_model=pickle.load(file)
    
st.sidebar.header('Input Paramaters')


BMI=st.number_input('BMI')
Cerv_Wid_cms=st.number_input('Cerv_Wid_cms')
Induction=st.number_input('Induction')
Position_Score=st.number_input('Position_Score')
Age=st.number_input('Age')
Total_Bishop_Score=st.number_input('Total_Bishop_Score')
Cerv_Len_cms=st.number_input('Cerv_Len_cms')
Ob_Score=st.number_input('Ob_Score')

if st.button('Predict Delivery Mode'):
    input_data=[BMI,
                Cerv_Wid_cms,
                Induction,
                Position_Score,
                Age,
                Total_Bishop_Score,
                Cerv_Len_cms,
                Ob_Score]
    column_names=['BMI', 'Cerv_Wid_cms', 'Induction', 'Position_Score','Age',
                  'Total_Bishop_Score', 'Cerv_Len_cms', 'Ob_Score']
    
    input_df=pd.DataFrame([input_data],columns=column_names)
    Delivery_Mode_predict=rfclassifier_model.predict(df_data)
    
    if Delivery_Mode_predict[0]==0:
        st.write(' The Predicted Delivery Mode is C-Section Delivery')
    else:
        st.write(' The Predicted Delivery Mode is Normal Delivery')

