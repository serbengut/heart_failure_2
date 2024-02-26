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
    page_title = 'Heart Failure & Precision - Recall',
    #page_icon = 'DS4B_Logo_Blanco_Vertical_FB.png',
    layout = 'wide')

st.title('Heart Failure & Precision - Recall')
#st.divider()
st.header('Introducir datos paciente:')

col1,col2,col3,col4 = st.columns(4)
with col1:

### vble 1:
#age = st.number_input(label='Edad:',step=1.,format="%.0f")

    ### vble 2:
    chest_pain_type = st.selectbox('Chest Pain Type', ['ATA', 'NAP', 'ASY','TA'])

### vble 3:
#cholesterol = st.slider('Cholesterol level', 70, 800)

### vble 4:
#exercise_angina = st.selectbox('Exercise Angina', ['Y', 'N'])

### vble 5:
#fasting_bs = st.selectbox('Fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]', [1, 0])

### vble 6:
#max_hr = st.slider('Max heart rate', 80, 300)

with col2:
    ### vble 7:
    oldpeak = st.number_input(label='OldPeak', min_value=-2.0, max_value=6.0,step=0.1,format="%.2f")
#oldpeak=float(oldpeak)

### vble 8:
#resting_bp = st.slider('Resting heart rate', 80, 200)

### vble 9:
#resting_ecg = st.selectbox('Resting ECG', ['Normal', 'ST','LVH'])

with col3:
    ### vble 10:
    sex = st.selectbox('Sex', ['M', 'F'])

with col4:
    ### vble 11:
    st_slope = st.selectbox('Slope of the peak exercise:', ['Up', 'Flat','Down']) 
st.divider()

registro = pd.DataFrame({'chest_pain_type':chest_pain_type,                                              
                         'sex':sex,
                         'oldpeak':oldpeak,
                         'st_slope':st_slope
                         }
                        ,index=[0])

colu1,colu2 = st.columns(2)

with colu1:
    st.write('Datos del paciente introducidos:')
    registro

with colu2:
    #umbral_usu = st.number_input(label='Definir el Umbral (entre 0 y 1)', min_value=0.01, max_value=0.99,step=0.01,format="%.2f")
    umbral_usu = 0.12
st.divider()

########################################################################################
def carga_x_y():
    pred = np.loadtxt('pred_final.txt')
    val_y_final = np.loadtxt('val_y_final_.txt')
    return pred, val_y_final
########################################################################################
########################################################################################
def calcular_metricas(real, scoring, umbral):
    
    #CALCULAR LA DECISION SEGUN EL UMBRAL
    predicho = np.where(scoring > umbral,1,0) 
    
    #CALCULAR TODAS LAS MÉTRICAS
    conf = confusion_matrix(real,predicho)

    tn, fp, fn, tp = conf.ravel()

    #total_casos = y.shape[0]
    
    #accuracy = (tn + tp) / total_casos
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    #F1 = 2 * (precision * recall) / (precision + recall)

    #IMPRIMIR RESULTADOS
    print('\nMatriz de confusión\n',pd.DataFrame(conf))
    #print('\naccuracy:',round(accuracy,3))
    print('\nprecision:',round(precision,5))
    print('\nrecall:',round(recall,5))
    #print('\nF1:',round(F1,3))
    #print(predicho)
    return precision, recall
########################################################################################
pred, val_y_final = carga_x_y()

precision, recall = calcular_metricas(val_y_final, pred, umbral_usu)

fallo = ejecutar_modelo(registro)

##### boton calcular prob fallo cardiaco: ##########################
#with st.sidebar:
    #st.image('heart.jpg')
####################################################################

column1,column2,column3,column4 = st.columns([1,2.3,1.5,1.5])

with column1:
    if st.sidebar.button('CALCULAR POSIBILIDAD DE FALLO CARDIACO y MAX ROI'):
        #fallo = ejecutar_modelo(registro)
        st.write('Probabilidad de fallo cardicaco:')
        st.title(f'{round(100*fallo,2)}%')
        

        pred, val_y_final = carga_x_y()

        precision, recall = calcular_metricas(val_y_final, pred, umbral_usu)

    else:
        st.write('DEFINE LOS PARÁMETROS Y HAZ CLICK EN CALCULAR POSIBILIDAD DE FALLO CARDIACO')

with column2:
    st.write('El umbral para decidir si aplicar o no TRATAMIENTO PREVENTIVO se ha fijado en:')
    st.subheader(f'{round(100*(umbral_usu),2)}%')

with column3:
    st.write(f'Con el umbral definido, la Precision obtenida es:')
    st.subheader(f'{round(100*(precision),2)}%')

with column4:
    st.write(f'Con el umbral definido, el Recall obtenido es:')
    st.subheader(f'{round(100*(recall),2)}%')
st.divider()
  
if fallo > umbral_usu:
    st.header('PACIENTE ELEGIBLE PARA HACER TRATAMIENTO PREVENTIVO')
else:
    st.header('PACIENTE NO ELEGIBLE PARA HACER TRATAMIENTO PREVENTIVO')






