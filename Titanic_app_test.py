#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 12:15:48 2021

@author: maximejacoupy
"""

# #######################################################################################################################
#                                              # === LIBRAIRIES === #
# #######################################################################################################################

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import numpy as np
from PIL import Image

# #######################################################################################################################
#                                              # === INIT === #
# #######################################################################################################################

name = None
s = None

# #######################################################################################################################
#                                              # === IMAGES === #
# #######################################################################################################################

img_personne = Image.open("images/Personne.jpg")
img_titanic = Image.open("images/Titanic.jpg")
img_homme = Image.open("images/Homme.png")
img_femme = Image.open("images/Femme.png")
img_autre = Image.open("images/Autre.png")
img_jack = Image.open("images/Jack.jpg")
img_rose = Image.open("images/Rose.jpg")

# #######################################################################################################################
#                                              # === DATA PREPARATION === #
# #######################################################################################################################

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')


train_df = train_df.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
train_df['Sex'] = train_df['Sex'].apply(lambda x: 1 if x == 'male' else 0)
test_df['Sex'] = test_df['Sex'].apply(lambda x: 1 if x == 'male' else 0)
train_df['Embarked'] = test_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
test_df['Embarked'] = test_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)


test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
train_df = train_df.dropna()
test_df = test_df.dropna()

# #######################################################################################################################
#                                              # === MACHINE LEARNING === #
# #######################################################################################################################

X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]

random_forest = RandomForestClassifier(n_estimators=10)
random_forest.fit(X_train, Y_train)

# #######################################################################################################################
#                                              # === STREAMLIT === #
# #######################################################################################################################

# 0. Créate sidebar
st.sidebar.image(img_personne, _______="Votre humble créateur de l'application")
st.sidebar._______("[Lien vers le GitHub du créateur](_____________)")
st._______.markdown("Cette application est basée sur un modèle de Machine Learning !")
st._______.markdown("Elle utilise un modèle de classification par Random Forest entrainé avec le légendaire jeu de données 'Titanic Survivor' de Kaggle.")
st._______.markdown("Ce modèle a un taux de bonne prédiction de 82,5% lorsqu'il s'agissait de prédire si une personne aurait réussi à monter sur un canot de sauvetage ou se serait noyé dans l'océan.")
st._______.markdown("J'espère que cela vous plaira et n'oubliez pas : **Les femmes et les enfants d'abord !**")

# #######################################################################################################################

# 1. Insert image
st._______(img_titanic)

# #######################################################################################################################

# 2. Affiche le titre
st._______("Survivrez-vous au Titanic ?")

# #######################################################################################################################

# 3. Entrer le nom
name = st._______("Quel est votre nom ?")
st.markdown("""---""")

# #######################################################################################################################

# BONUS

# Bonus 1. Drag and drop photo

# Bonus 2. Affiche photo en miniature

# Bonus 3. Estime age et genre

# #######################################################################################################################

# 4. Choisir le genre
col1, col2, col3 = st._______(____)

with col1:
    st._______(img_homme)
    male = st._______("Homme")
    if male:
        s = 1

with col2:
    st._______(img_femme)
    female = st._______("Femme")
    if female:
        s = 0

with col3:
    st._______(img_autre)
    other = st._______("Non binaire")
    if other:
        s = 0.5
st.markdown("""---""")

# #######################################################################################################################

# 5. Selectionner l'age
age = st.number_input("_______", _______=1, _______=1, _______=120, _______=20)
st.markdown("""---""")

# #######################################################################################################################

# 6. Selectionner la classe
Class = st._______("Dans quelle classe étiez-vous lors de votre dernier voyage ?", ['_______', '_______', '_______'], index=0)
c = 0
if _______ == "Affaire":
    c = 1
elif _______ == "Business":
    c = 2
elif _______ == "Economique":
    c = 3
st.markdown("""---""")

# #######################################################################################################################

# 7.1 Etes vous mariés
celib = st._______(_______)
if celib == "Oui":
    m = 0
else:
    m = 1
st.markdown("""---""")

# #######################################################################################################################

# 8. Voyagez vous avec vos frères et soeur ?
fs = st._______("Combien de vos frères et/ou soeurs voyagent avec vous d'habitude ?", _______=1, _______=0)
st.markdown("""---""")

# #######################################################################################################################

# 9. Est ce que vous voyager avec vos enfants ?
enfants = st._______("Avez-vous des enfants ?", ["Non", "Oui"], _______=0)
if enfants == 'Oui':
    # Si oui, combien ?
    nb_enfants = st.number_input("Combien voyagent avec vous ?", _______=1, _______=0)
else:
    nb_enfants = 0
st.markdown("""---""")

# #######################################################################################################################

# 10. D'ou etes vous partis ?
Port = st._______("D'où souhaitez-vous partir ?", ["_______", "_______", "_______"])
if _______ == "France":
    p = 1
elif _______ == "Nord du Royaume-Uni":
    p = 2
elif _______ == "Sud du Royaume-Uni":
    p = 0
st.markdown("""---""")

# #######################################################################################################################

# 11. Generalement est ce que vous achetez vos billets en avance ou au dernier moment ?
Quand = st._______("Quand achetez-vous généralement vos billets ?", ["Dés que possible", "Quelques semaines avant le depart", "Au dernier moment"])
if Quand == "_______":
    f = X_train.loc[X_train['Pclass'] == c]['Fare'].quantile([0.25])
elif Quand == "_______":
    f = X_train.loc[X_train['Pclass'] == c]['Fare'].quantile([0.5])
elif Quand == "_______":
    f = X_train.loc[X_train['Pclass'] == c]['Fare'].quantile([0.75])
st.markdown("""---""")
    
# #######################################################################################################################

# 12. Creation Dataframe
button = st._______("_______")
if name and _______:
    try:
        dataframe = pd.DataFrame(np.array([[c, s, age, (m+fs), nb_enfants, f, p]]), columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])    
        pred = random_forest.predict(dataframe)[0]
        
        if pred == 0:
            st._______("Désolé, "+name+", vous n'auriez probablement pas survécu au naufrage du titanic")
            st.image(img_jack)
        elif pred == 1:
            st._______("Vous avez de la chance "+name+", vous auriez probablement survecu au naufrage du titanic")
            st.image(img_rose)
            st.balloons()
    except Exception as e:
        if s is None:
            st._______("Veuillez sélectionner un genre")
elif button:
    st._______("Veuillez renseigner votre nom")
            
    
# #######################################################################################################################

#                                          # === END OF FILE === #

# #######################################################################################################################
