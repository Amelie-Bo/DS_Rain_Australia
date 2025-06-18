# Contenu de streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Charger les données
df = pd.read_csv("data/weatherAUS.csv")

# Load X_test with error handling
try:
    df_new_data = pd.read_csv("data/data_2024-25.csv") #Essayer 2024_25 pour voir si ok
    st.success("data_2024-25.csv loaded successfully!")
except FileNotFoundError:
    st.error("Error: data_2024-25.csv not found. Please ensure the file is in the correct directory.")
except Exception as e:
    st.error(f"An error occurred while loading data_2024-25.csv: {e}")

# 2. Définir la structure   
st.title("Projet de classification binaire sur la pluie en Australie")
st.sidebar.title("Sommaire")
pages=["Exploration", "DataVizualization", "Modélisation","Evaluation"]
page=st.sidebar.radio("Aller vers", pages)

# 3. Sur la page Exploration 
if page == pages[0] : 
  st.write("### Introduction")
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
  # Analyse multivariée par matrice de corrélation
  fig, ax = plt.subplots()
  sns.heatmap(df.corr(), ax=ax)
  st.write(fig)

# 4. Sur la page Modélisation 
if page == pages[2] :
  st.write("### Modélisation")# sur X_test preprocesse ou non?(mon preprocessing + modelisationprend qq minutes )

  # Supprimer les variables inutiles
  X_test = X_test.drop(['Evaporation', 'Sunshine'], axis=1)

if page == pages[3] :
  st.write("### Evaluation") #
