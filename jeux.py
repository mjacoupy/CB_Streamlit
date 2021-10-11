#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 12:34:54 2021

@author: maximejacoupy
"""
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from datetime import date



st.title("Jeux Career Booster Streamlit")
st.markdown("""---""")



jeu = st.sidebar.selectbox('', ['Age', 'Carte', 'IMC', 'Iris'])

if jeu == 'Age':
    st.header('Age')
    
    nom = st.text_input("Quel est voter prénom")
    naissance = st.date_input("Quel est votre jour de naissance", date(1988, 3, 12))
    aujourdhui = date.today()
    
    time_difference = aujourdhui - naissance
    age = int(time_difference.days/365.25)
    
    button = st.button("Calcul de l'age")
    
    if age and button:
        st.markdown("**"+nom+"** a **"+str(age)+"** ans")
        
        
if jeu == 'Carte':     
    st.header('Carte')

    debut = st.date_input("Date de disponibilité de debut")
    fin = st.date_input("Date de disponibilité de fin")

    villes = st.multiselect("Ville", ('Paris', 'Marseille', 'Lille', 'Toulouse', 'Bordeaux', 'Nantes'), default=['Paris'])

    Paris = [48.85, 2.35]
    Marseille = [43.3, 5.38]
    Lille = [50.63, 3.06]
    Toulouse = [43.6, 1.44]
    Bordeaux = [44.84, -0.58]
    Nantes = [47.22, -1.55]

    lat = []
    lon = []
    for v in villes:
        if v == 'Paris':
            lat.append(Paris[0])
            lon.append(Paris[1])
        elif v == 'Marseille':
            lat.append(Marseille[0])
            lon.append(Marseille[1])
        elif v == 'Lille':
            lat.append(Lille[0])
            lon.append(Lille[1])
        elif v == 'Toulouse':
            lat.append(Toulouse[0])
            lon.append(Toulouse[1])
        elif v == 'Bordeaux':
            lat.append(Bordeaux[0])
            lon.append(Bordeaux[1])
        elif v == 'Nantes':
            lat.append(Nantes[0])
            lon.append(Nantes[1])

    data = {'lat': lat, 'lon': lon}
    df = pd.DataFrame(data)

    st.map(df, zoom=4)
        
    
    

if jeu == 'IMC':
    st.header('IMC')

    poids = st.number_input("Quel est votre poids en kg ?")
    taille = st.number_input("Quelle est votre taille en cm ?")
    
    button = st.button("Calcul de l'IMC")
    
    if poids and taille and button:
        bmi = poids / ((taille/100)**2) 
        
        if bmi < 18.5: 
            st.warning("You are Underweight") 
            st.image("/Users/maximejacoupy/Developments/CB_Streamlit/input/BMI_1.png")
        elif(bmi >= 18.5 and bmi < 25): 
            st.success("Normal")    
            st.image("/Users/maximejacoupy/Developments/CB_Streamlit/input/BMI_2.png")
        elif(bmi >= 25 and bmi < 30): 
            st.warning("Overweight") 
            st.image("/Users/maximejacoupy/Developments/CB_Streamlit/input/BMI_3.png")
        elif(bmi >= 30 and bmi < 35): 
            st.error("Obese") 
            st.image("/Users/maximejacoupy/Developments/CB_Streamlit/input/BMI_4.png")
        elif(bmi >= 35): 
            st.error("Extremly Obese") 
            st.image("/Users/maximejacoupy/Developments/CB_Streamlit/input/BMI_5.png")
            
            
if jeu == 'Iris':
    st.header('Iris')
    
    # Iris Setosa, Iris Versicolour, Iris Virginica
    name = st.radio("Plante", ["Iris Virginica", "Iris Setosa", "Iris Versicolour"])
    
    if name:
        
        if name == "Iris Setosa":
            n = 0
        elif name == "Iris Versicolour":
            n = 1
        elif name == "Iris Virginica":
            n = 2
    
    
        data = datasets.load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        
        sub_df = df[df['target'] == n]
            
        g = plt.figure()
        
        sns.boxplot(data=sub_df.iloc[:, :4], orient="v", palette="Set2").set_title("Valeurs pour la plante "+name)
        
        st.pyplot(g)            