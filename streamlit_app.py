# Contenu de streamlit_app.py

## Chargement des librairies
import streamlit as st

import pandas as pd
import numpy as np
from scipy.stats import randint

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, f1_score, classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, 
                             roc_curve, precision_recall_fscore_support, brier_score_loss)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.decomposition import PCA

from imblearn.metrics import classification_report_imbalanced 
import xgboost as xgb

import pickle
from joblib import load
import cloudpickle

import os

import streamlit-shap
from streamlit_shap import st_shap
import shap

# 1. Charger les données
df = pd.read_csv("data/weatherAUS.csv")
X_test = pd.read_csv("data/data_2024-25.csv")

# 2. Définir la structure
st.title("Projet de classification binaire sur la pluie en Australie") # sera répercuté sur toutes les pages du Streamlit
st.sidebar.title("Sommaire")
pages=["Exploration", "DataVizualization", "Modélisation","Evaluation"]
page=st.sidebar.radio("Aller vers", pages)

# 3. Sur la page Exploration
if page == pages[0] :
  st.header("Introduction") 
  st.write("Ce projet traite... ") #\n n'apporte rien autant faire un nouveau st.write
  
  # test visuel
  st.title("st.titre") #Gras, 44px
  st.header("st.header") #36px semi gras, comme ##
  st.subheader("st.subheader") #28px semi gras, comme ### 
  st.write("#### 4#..") #24px semi gras 
  st.write("##### 5#..") #20px semi gras 
  st.write("###### 6#..") #16px semi gras 
  st.write("normal") #16px normal (comme un print dans une jupyer NB)
  st.markdown("Texte **gras**, *italique*, et un [lien](https://example.com)") #pour mettre en forme
  

  st.dataframe(df.head(10))
  st.write(df.shape) #equivalent de print
  st.dataframe(df.describe()) #st.dataframe pour appeler des méthodes pandas qui entraine un affichage de df

  if st.checkbox("Afficher les NA") : #quand on coche la case, on affiche la méthode ci-dessous
    st.dataframe(df.isna().sum())

# 3. Sur la page Datavizualisation
#Sur le fichier source? sur le nouveau jeu de données?
if page == pages[1] :
  st.write("### DataVizualization")
  #Afficher un graphique de la variable cible "Pluie demain"
  fig = plt.figure()
  sns.countplot(x = 'RainTomorrow', data = df)
  st.pyplot(fig)
  
  # Impact de features sur la variable cible
  fig1 = plt.figure()
  sns.countplot(x = 'RainTomorrow', hue='RainToday', data = df, title = "lien Variable cible et RainToday")
  st.pyplot(fig1)
  
  fig2 = sns.catplot(x='Cloud3pm', y='RainTomorrow', data=df, kind='point')
  st.pyplot(fig2)
  
  fig3 = sns.lmplot(x='Temp3pm', y='RainTomorrow', hue="Cloud3pm", data=df)
  st.pyplot(fig3)

  # Analyse multivariée par matrice de corrélation
  fig, ax = plt.subplots()
  sns.heatmap(df.corr(), ax=ax)
  st.write(fig)

  #Plotly
  # fig4 = px.scatter(df, x=, y=, title="")
  st.plotly_chart(fig4) #ADD

# 4. Sur la page Modélisation
if page == pages[2] :
  st.write("### Modélisation")# sur X_test preprocesse ou non?(mon preprocessing + modelisationprend qq minutes )

  # Supprimer les variables inutiles
  X_test = X_test.drop(['Evaporation', 'Sunshine'], axis=1)

if page == pages[3] :
  st.write("### Evaluation") #
