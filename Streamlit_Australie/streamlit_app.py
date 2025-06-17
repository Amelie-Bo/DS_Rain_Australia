# Contenu de strealit_app.py 
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Charger les données

df = pd.read_csv("weatherAUS.csv")
X_test =  pd.read_csv("data_2024-25.csv")

st.title("Projet de classification binaire sur la pluie en Australie")
st.sidebar.title("Sommaire")
pages=["Exploration", "DataVizualization", "Modélisation","Evaluation"]
page=st.sidebar.radio("Aller vers", pages)

if page == pages[0] : #Sur la page Exploration
  st.write("### Introduction")
  st.dataframe(df.head(10))
  st.write(df.shape) #equivalent de print
  st.dataframe(df.describe()) #st.dataframe pour appeler des méthodes pandas qui entraine un affichage de df

  if st.checkbox("Afficher les NA") : #quand on coche la case, on affiche la méthode ci-dessous
    st.dataframe(df.isna().sum())


if page == pages[1] :
  st.write("### DataVizualization") #Sur le fichier source? sur le nouveau jeu de données?
  #Afficher un graphique de la variable cible "Pluie demain"
  fig = plt.figure()
  sns.countplot(x = 'RainTomorrow', data = df)
  st.pyplot(fig)
  # Analyse multivariée par matrice de corrélation
  fig, ax = plt.subplots()
  sns.heatmap(df.corr(), ax=ax)
  st.write(fig)

if page == pages[2] :
  st.write("### Modélisation")# sur X_test preprocesse ou non?(mon preprocessing + modelisationprend qq minutes )

  # Supprimer les variables inutiles
  X_test = X_test.drop(['Evaporation', 'Sunshine'], axis=1)

if page == pages[3] :
  st.write("### Evaluation") #
