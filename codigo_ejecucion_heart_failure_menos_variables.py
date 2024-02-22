import numpy as np
import pandas as pd
import cloudpickle

#import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split

#Automcompletar rápido
#%config IPCompleter.greedy=True

from janitor import clean_names

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

import seaborn as sns
from sklearn.metrics import confusion_matrix



def ejecutar_modelo(df):
   
    nombre_pipe_ejecucion = 'pipe_ejecucion.pickle'

    with open(nombre_pipe_ejecucion, mode='rb') as file:
        pipe_ejecucion = cloudpickle.load(file)

    scoring = pipe_ejecucion.predict_proba(df)[:,1]

    return 'El porcentaje de tener fallo cardiaco es: {}%'.format(round(scoring[0]*100,2))

















