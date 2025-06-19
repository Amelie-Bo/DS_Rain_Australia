#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
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

import streamlit_shap
from streamlit_shap import st_shap
import shap

import pickle
from joblib import load
import cloudpickle

import os
import time
import requests
from io import StringIO
from datetime import datetime
from dateutil.relativedelta import relativedelta

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 1. Charger les données
df = pd.read_csv("data/weatherAUS.csv")
liste_colonne_df = df.columns #servira à ordonner les colonnes dans les données actuelles
X_test = pd.read_csv("data/data_2024-25.csv")
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 2. Définir la structure
st.title("Projet de classification binaire sur la pluie en Australie") # sera répercuté sur toutes les pages du Streamlit
st.sidebar.title("Sommaire")
pages=["Exploration", "DataVizualization", "Modélisation","Evaluation","Datas actuelles"]
page=st.sidebar.radio("Aller vers", pages)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
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
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 3. Sur la page Datavizualisation
#Sur le fichier source? sur le nouveau jeu de données?
if page == pages[1] :
  st.header("DataVizualization")
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
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 4. Sur la page Modélisation
if page == pages[2] :
  st.header("Modélisation")# sur X_test preprocesse ou non?(mon preprocessing + modelisationprend qq minutes )

  # Supprimer les variables inutiles
  X_test = X_test.drop(['Evaporation', 'Sunshine'], axis=1)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
if page == pages[3] :
  st.header("Evaluation")
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
if page == pages[4] :
  st.header("Datas actuelles")
#-1. Collecte des données actuelles---------------------------------------------------------------------------------------------------------------------------------
  #1.1 Initialisation
  ##1.1.1 Liste des mois
  mois_courant = datetime.now().replace(day=1)
  mois_depart = mois_courant - relativedelta(months=1) # Mois de départ = mois précédent
  # Générer les 13 mois vers le passé (de -12 à 0 mois avant mois_depart)
  liste_mois_a_selectionner = [(mois_depart - relativedelta(months=i)).strftime("%Y%m") for i in reversed(range(13))]

  ##1.1.2 Dico stations
  with open("dico_scaler/dico_station.pkl", "rb") as fichier:
    dico_stations_BOM = pickle.load(fichier)

  ##1.1.3 Noms des colonnes dans le df généré (futur X_test)
  nom_colonnes_df_principal = {"Minimum temperature (°C)" : "MinTemp",
                             "Maximum temperature (°C)": "MaxTemp",
                             "Rainfall (mm)" : "Rainfall",
                             "Evaporation (mm)" : "Evaporation",
                             "Sunshine (hours)" :"Sunshine",
                             "Direction of maximum wind gust ":"WindGustDir",
                             "Speed of maximum wind gust (km/h)" : "WindGustSpeed",
                             "Time of maximum wind gust" : "Time of maximum wind gust" ,  #nouvelle colonne
                             "9am Temperature (°C)": "Temp9am",
                             "9am relative humidity (%)" : "Humidity9am",
                             "9am cloud amount (oktas)":"Cloud9am",
                             "9am wind direction" :"WindDir9am",
                             "9am wind speed (km/h)": "WindSpeed9am",
                             "9am MSL pressure (hPa)":"Pressure9am",
                             "3pm Temperature (°C)":"Temp3pm",
                             "3pm relative humidity (%)":"Humidity3pm",
                             "3pm cloud amount (oktas)": "Cloud3pm",
                             "3pm wind direction" : "WindDir3pm",
                             "3pm wind speed (km/h)": "WindSpeed3pm",
                             "3pm MSL pressure (hPa)":"Pressure3pm"}

  # 1.2 Saisie utilisateur
  st.subheader("Sélection ")

  liste_mois = st.multiselect("Sélectionnez un ou plusieurs mois", liste_mois_a_selectionner)

  ## 1.2.1 Afficher le nom des stations dans la liste déroulante
  # Multiselect avec affichage du nom de la station
  stations_selectionnees = st.multiselect(
      "Sélectionnez une ou plusieurs stations",
      options=list(dico_stations_BOM.keys()),
      format_func=lambda x: dico_stations_BOM[x][2])
  # Générer un dictionnaire filtré identique en format à l’original
  dico_stations_DWO = {
      k: dico_stations_BOM[k]
      for k in stations_selectionnees}

  # 1.2.2 Tant que l'utilisateur n'a pas fait de sélection complète (mois + location) ne pas aller plus loin
  if not liste_mois or not dico_stations_DWO:
      st.stop()

  # 2 Code récupérant les csv et les conslidant dans le Df df_conso_station à partir d'une liste d'url
  compteur = 0 #pour consolider les df de station dans un un seul df : df_conso_station
  df_conso_station=pd.DataFrame()

  for no_report in dico_stations_DWO :
      i=0
      compteur+= 1
      df_une_station=pd.DataFrame()

      for le_annee_mois in liste_mois :
          url_concatene = ("http://www.bom.gov.au/climate/dwo/"+str(le_annee_mois)+"/text/"+no_report+"."+le_annee_mois+".csv") # Exemple : http://www.bom.gov.au/climate/dwo/202412/text/IDCJDW2804.202412.csv
          i+= 1
          # Essayer de télécharger le fichier avec des en-têtes de type navigateur
          headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
          # Effectuer la requête
          response = requests.get(url_concatene, headers=headers)
          # Vérifier si la requête est réussie
          if response.status_code == 200:
              # Utiliser StringIO pour lire le texte CSV dans un DataFrame
              csv_data = StringIO(response.text)
              # L'entete n'est a=pas toujours sur la meme ligne : On lit le fichier ligne par ligne pour trouver le header qui commence par ,"Date",
              lines = response.text.splitlines()
              # Trouver la ligne où "Date" apparaît pour la première fois
              header_row = None
              for i, line in enumerate(lines):
                  if "Date" in line:
                      header_row = i
                      break
              df_recupere = pd.read_csv(csv_data, sep=",", skiprows=header_row, encoding="latin1")
              #Faire un df consolidé par station
              if i == 1 :
                  df_une_station = df_une_station
              else :
                  df_une_station = pd.concat([df_une_station, df_recupere], ignore_index=True)
          else:
              st.write("Erreur lors du chargement de l'URL pour {} de {} : {} - URL: {}".format(le_annee_mois, dico_stations_DWO[no_report][2], response.status_code,url_concatene))

    # 2.2 Mise en forme du csv collecté
      df_une_station = df_une_station.rename(nom_colonnes_df_principal, axis = 1) #Mettre les noms de colonnes du df principal
      df_une_station = df_une_station.drop(["Unnamed: 0","Time of maximum wind gust"],axis =1) #Suppression de colonnes

      #Ajout de colonnes
      #Insérer en 2e position (loc=1) le nom de la station (3e colonne du dico) associé à ce rapport dans le dictionnaire dico_stations_DWO
      df_une_station.insert(1,column="Location", value=dico_stations_DWO[no_report][2]) #

      df_une_station["RainToday"]=df_une_station["Rainfall"].apply(lambda x: "Yes" if x>1 else "No")
      #Met les valeurs de RainToday à l'indice précédent dans RainTomorrow
      df_une_station["RainTomorrow"] = np.roll(df_une_station["RainToday"].values, 1) #Ex RainToday 02/01/24 -> RainTomorrow 01/01/2024

      #Supprimer le dernier relevé du df car RainTomorrow y sera toujours inconnu (suite au np.roll c'est la valeur de RainToday a la 1e ligne du df, et il aurait fallu celle du lendemain de la dernier ligne du df).
      df_une_station.drop(df_une_station.index[-1], inplace=True)

      #Mettre les colonnes dans le même ordre que le df de départ du projet
      df_une_station = df_une_station[liste_colonne_df]


      #Faire un df consolidé (df_conso_station) des df unitaire par station (df_une_station)
      if compteur == 1 :
          df_conso_station = df_une_station
      else :
          df_conso_station = pd.concat([df_conso_station, df_une_station], ignore_index=True)


  #-2. Preprocessing de base---------------------------------------------------------------------------------------------------------------------------------
  df_X_test = df_conso_station

  ## 2.1 Modification de la vitesse "Calm" par 0km/h
  df_X_test["WindSpeed9am"] = df_X_test["WindSpeed9am"].apply(lambda x: 0 if x =="Calm" else x)
  df_X_test["WindSpeed3pm"] = df_X_test["WindSpeed3pm"].apply(lambda x: 0 if x =="Calm" else x)
  df_X_test["WindGustSpeed"] = df_X_test["WindGustSpeed"].apply(lambda x: 0 if x =="Calm" else x)

  ## 2.2 Suprresion 25% des NAN
  total_cells_per_location = df_X_test.groupby("Location").size() * (df_X_test.shape[1] - 1)  # -1 pour exclure la colonne Location elle-même
  nan_counts_per_location = df_X_test.drop(columns="Location").isna().groupby(df_X_test["Location"]).sum().sum(axis=1)
  nan_ratio = nan_counts_per_location / total_cells_per_location
  valid_locations = nan_ratio[nan_ratio <= 0.25].index
  df_X_test = df_X_test[df_X_test["Location"].isin(valid_locations)]

  ## 2.3 Ajout de la latitude et de la longitude
  with open("dico_scaler/dico_station_geo.pkl", "rb") as fichier:
    dico_charge = pickle.load(fichier)
  df_dico_station_geo = pd.DataFrame.from_dict(dico_charge, orient="index",columns=["Lat", "Lon"])
  df_dico_station_geo.columns = ["Latitude", "Longitude"]
  df_X_test = df_X_test.merge(right=df_dico_station_geo, left_on="Location", right_index=True, how="left")

  ## 2.3 Date, Saison
  df_X_test["Date"]=pd.to_datetime(df_X_test["Date"], format = "%Y-%m-%d")
  df_X_test["Month"] = df_X_test['Date'].dt.month
  df_X_test["Year"] = df_X_test['Date'].dt.year
  df_X_test["Saison"] = df_X_test["Month"].apply( lambda x : "Eté" if x in [12, 1, 2] else "Automne" if x in [3, 4, 5] else "Hiver" if x in [6, 7, 8] else "Printemps")

  ## 2.4 Ajout du climat
  climat_mapping = pd.read_csv("dico_scaler/climat_mapping.csv", index_col="Location")
  climat_mapping_series = climat_mapping.squeeze()  # Convertir en Series pour faciliter le mapping
  df_X_test['Climat'] = df_X_test["Location"].map(climat_mapping_series) #pour chaque valeur de df.Location, on récupère la valeur correspondante dans climat_mapping

  ## 2.5 Suppression des features
  df_X_test = df_X_test.drop(["Sunshine","Evaporation"], axis = 1)

  ## 2.6 Traitement de la variable cible : Suppression des NaN et Label Encoder
  df_X_test = df_X_test.dropna(subset=["RainTomorrow"], axis=0, how="any")
  encoder=LabelEncoder()
  df_X_test["RainTomorrow"] = encoder.fit_transform(df_X_test["RainTomorrow"])  #N=0, Y=1

  # 3 Sélectionner un jour (avant d'enlever Date)---------------------------------------------------------------------------------------------------------------------------------
  st.dataframe(df_X_test.head(10))
  #pb pour ouvrir dico station et choisir
  # 4 Reste du preprocessing (Date supprimée) ##stop ici

  # 5 Prédiction------------------------------------------------------------------------------------------------------------------------------------------------------------------
  # 6 Evaluation------------------------------------------------------------------------------------------------------------------------------------------------------------------
  # 7 Interprétation--------------------------------------------------------------------------------------------------------------------------------------------------------------
