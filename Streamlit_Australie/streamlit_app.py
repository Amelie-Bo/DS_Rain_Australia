import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Charger les données
# __file__ pointe vers streamlit_app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "weatherAUS.csv"))
X_test =  pd.read_csv(os.path.join(BASE_DIR, "data_2024-25.csv")) 

st.title("Projet de classification binaire sur la pluie en Australie")
st.sidebar.title("Sommaire")
pages=["Exploration", "DataVizualization", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)

if page == pages[0] : #Sur la page Exploration
  st.write("### Introduction")
  st.dataframe(df.head(10))
  st.write(df.shape) #equivalent de print
  st.dataframe(df.describe()) #st.dataframe pour appeler des méthodes pandas qui entraine un affichage de df
  
  if st.checkbox("Afficher les NA") : #quand on coche la case, on affiche la méthode ci-dessous
    st.dataframe(df.isna().sum()) 


if page == pages[1] : 
  st.write("### DataVizualization")
  #Afficher un graphique de la variable cible "Plui demain"
  fig = plt.figure()
  sns.countplot(x = 'RainTomorrow', data = df)
  st.pyplot(fig)
  # Analyse multivariée par matrice de corrélation
  fig, ax = plt.subplots()
  sns.heatmap(df.corr(), ax=ax)
  st.write(fig)
  
  from src_py.visualization import map_rainfall_by_station
    
    @st.cache_data
    def load_my_data():
        # Exemple de chargement ; à adapter
        df = pd.read_csv("data/mon_dataset.csv")
        # … tout pré-traitement nécessaire …
        return df
    
    @st.cache_data
    def load_localisations():
        # Exemple : charger vos coords depuis un JSON ou CSV
        # Ici on hardcode juste un dict
        return {
          "Station A": (-33.86, 151.21),
          "Station B": (-37.81, 144.96),
          # …
        }
    
    def main():
        st.title("Carte des précipitations moyennes")
    
        df = load_my_data()
        localisations = load_localisations()
    
        st.write("Aperçu des données :", df.head())
    
        # On récupère la figure depuis notre module
        fig = map_rainfall_by_station(df, localisations)
    
        # On l'affiche avec Streamlit
        st.plotly_chart(fig, use_container_width=True)
    
    if __name__ == "__main__":
        main()

if page == pages[2] : 
  st.write("### Modélisation")# sur X_test

  # Supprimer les variables inutiles
  df = df.drop(['Evaporation', 'Sunshine'], axis=1) 
