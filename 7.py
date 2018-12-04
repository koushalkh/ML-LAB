from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.models import BayesianModel
import numpy as np
from urllib.request import urlopen
import urllib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl
import pandas as pd
import pgmpy
Cleveland_data_URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data'
np.set_printoptions(threshold=np.nan)  
names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
         'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'heartdisease']
heartDisease = pd.read_csv(urlopen(Cleveland_data_URL),names=names) 
heartDisease.head()
print(heartDisease.head())
del heartDisease['ca']
del heartDisease['slope']
del heartDisease['thal']
del heartDisease['oldpeak']
heartDisease = heartDisease.replace('?', np.nan)
heartDisease.dtypes
print(heartDisease.dtypes)
heartDisease.columns
model = BayesianModel([('age', 'trestbps'), ('age', 'fbs'), ('sex', 'trestbps'), ('sex', 'trestbps'),
                       ('exang', 'trestbps'), ('trestbps',
                                               'heartdisease'), ('fbs', 'heartdisease'),
                       ('heartdisease', 'restecg'), ('heartdisease', 'thalach'), ('heartdisease', 'chol')])
model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)
print(model.get_cpds('age'))
print(model.get_cpds('chol'))
print(model.get_cpds('sex'))
model.get_independencies()
HeartDisease_infer = VariableElimination(model)
q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'age': 28})
print(q['heartdisease'])
q = HeartDisease_infer.query(
    variables=['heartdisease'], evidence={'chol': 100})
print(q['heartdisease'])