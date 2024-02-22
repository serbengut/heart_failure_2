import pandas as pd

import streamlit as st
#from streamlit_echarts import st_echarts
from codigo_ejecucion_heart_failure_menos_variables import *
import numpy as np
import cloudpickle
import pickle

from janitor import clean_names

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import HistGradientBoostingClassifier

from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

st.set_page_config(
     page_title = 'Heart Failure',
     #page_icon = 'DS4B_Logo_Blanco_Vertical_FB.png',
     #layout = 'wide')
)

st.title('Heart Failure')


### vble 1:
#age = st.number_input(label='Edad:',step=1.,format="%.0f")

### vble 2:
chest_pain_type = st.selectbox('Chest Pain Type', ['ATA', 'NAP', 'ASY','TA'])

### vble 3:
cholesterol = st.slider('Cholesterol level', 70, 800)

### vble 4:
exercise_angina = st.selectbox('Exercise Angina', ['Y', 'N'])

### vble 5:
#fasting_bs = st.selectbox('Fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]', [1, 0])

### vble 6:
max_hr = st.slider('Max heart rate', 80, 300)

### vble 7:
oldpeak = st.number_input(label='OldPeak', min_value=-2.0, max_value=6.0,step=0.1,format="%.2f")
#oldpeak=float(oldpeak)

### vble 8:
#resting_bp = st.slider('Resting heart rate', 80, 200)

### vble 9:
#resting_ecg = st.selectbox('Resting ECG', ['Normal', 'ST','LVH'])

### vble 10:
#sex = st.selectbox('Sex', ['M', 'F'])

### vble 11:
st_slope = st.selectbox('Slope of the peak exercise:', ['Up', 'Flat','Down']) 


registro = pd.DataFrame({'chest_pain_type':chest_pain_type,                                              
                         'cholesterol':cholesterol,
                         'exercise_angina':exercise_angina,
                         'max_hr':max_hr,
                         'oldpeak':oldpeak,
                         'st_slope':st_slope
                         }
                        ,index=[0])

registro

if st.sidebar.button('CALCULAR POSIBILIDAD DE FALLO CARDIACO'):

    fallo = ejecutar_modelo(registro)

    fallo

else:
    st.write('DEFINE LOS PARÁMETROS Y HAZ CLICK EN CALCULAR POSIBILIDAD DE FALLO CARDIACO')


