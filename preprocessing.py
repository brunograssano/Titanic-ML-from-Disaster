import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

def normalize(numericalData):
    return (numericalData - numericalData.mean()) / numericalData.std()

def ordinalEncoder(datos_a_codificar):
    encoder = OrdinalEncoder()
    return encoder.fit_transform(datos_a_codificar)

def oneHotEncoder(datos_a_codificar):
    encoder = OneHotEncoder(drop='first', sparse=False)
    return encoder.fit_transform(datos_a_codificar)

def encodeDataset(titanic_df,categorical_columns,numerical_columns):
    categoricalDataEncoded = oneHotEncoder(titanic_df[categorical_columns])
    numericalData = titanic_df[numerical_columns]
    data = np.hstack((np.array(numericalData), categoricalDataEncoded))
    return data

def encodeAndNormalizeData(titanic_df,categorical_columns,numerical_columns):
    categoricalDataEncoded = oneHotEncoder(titanic_df[categorical_columns])
    numericalData = titanic_df[numerical_columns]
    numericalData = normalize(numericalData)
    data = np.hstack((np.array(numericalData), categoricalDataEncoded))
    return data