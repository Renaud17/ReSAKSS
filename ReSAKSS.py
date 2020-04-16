import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual


from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic

#data ReSAKSS
names = ["year", "county_1", "county_2", "LST_day", "LST_night", "NDVI_min", "NDVI_avg", "NDVI_max", "rainfall_min", "rainfall_avg", "rainfall_max", "millet_ha", "corn_ha", "sorghum_ha", "rice_ha", "groundnuts_ha", "millet_tons", "corn_tons", "sorghum_tons", "rice_tons", "groundnuts_tons", "population"]
df = pd.read_excel("Dataset_ReSAKSS_Data_Challenge_FINAL.xlsx", skiprows=0, header=1, names=names)

df.head()

"""#**Data cleaning**"""

df.isnull().sum()

df.info()

#Imputation par la moyenne dans les colonnes millet_ha, sorghum_ha, rice_ha et groundnuts_ha
def imput_avg(x):
    avg = x.astype('float').mean(axis = 0)
    return x.replace(np.nan, avg, inplace = True)

for colonne in ['millet_ha', 'sorghum_ha', 'rice_ha', 'groundnuts_ha']:
    imput_avg(df[colonne])
    
#Suppression de lignes avec données manquantes au niveau des variables cibles
df.dropna(subset = ['millet_tons', 'corn_tons', 'sorghum_tons', 'rice_tons', 'groundnuts_tons'], axis = 0, inplace = True)

#Réinitialisation des indices
df.reset_index(drop = True, inplace = True)

df.info()

df1=df[['LST_day', 'LST_night', 'NDVI_min',
       'NDVI_avg', 'NDVI_max', 'rainfall_min', 'rainfall_avg', 'rainfall_max',
       'millet_ha', 'corn_ha', 'sorghum_ha', 'rice_ha', 'groundnuts_ha',
       'millet_tons', 'corn_tons', 'sorghum_tons', 'rice_tons',
       'groundnuts_tons', 'population']]




#Predictor columns
X=df1[['LST_day', 'LST_night', 'NDVI_min', 'NDVI_avg', 'NDVI_max',
       'rainfall_min', 'rainfall_avg', 'rainfall_max', 'millet_ha', 'corn_ha',
       'sorghum_ha', 'rice_ha', 'groundnuts_ha','population']]

#Outcome column
y = df1[['millet_tons', 'corn_tons',
       'sorghum_tons', 'rice_tons', 'groundnuts_tons']]






endog=df[['millet_tons', 'corn_tons',
       'sorghum_tons', 'rice_tons', 'groundnuts_tons']]
       
exog_coint=df[['NDVI_min', 'NDVI_avg', 'NDVI_max',
       'rainfall_min', 'rainfall_avg', 'rainfall_max', 'millet_ha', 'corn_ha',
       'sorghum_ha', 'rice_ha', 'groundnuts_ha']] 
       

exog=df[['LST_day', 'LST_night','population','millet_tons', 'corn_tons',
       'sorghum_tons', 'rice_tons', 'groundnuts_tons']]




#model
from statsmodels.tsa.vector_ar.vecm import VECM
vecm = VECM(endog = endog,exog=exog,exog_coint=exog_coint, k_ar_diff = 1, coint_rank = 5, deterministic ='cili')
vecm_fit = vecm.fit()



import pickle
with open('VECM_result.pkl', 'wb') as f:
  pickle.dump(vecm_fit ,f)





pip install git+https://github.com/ml-libs/mlserve.git

import mlserve
import json
from mlserve import build_schema

data_schema = mlserve.build_schema(df1)
with open('ReSAKSS.json', 'w') as f:
    json.dump(data_schema, f)
    

    
