#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 14:09:10 2021

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
from deepface import DeepFace
import cv2
import os

# #######################################################################################################################
#                                              # === INIT === #
# #######################################################################################################################
name = None
s = None
age = None
genre = None

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
st.sidebar.image(img_personne, caption="Votre humble créateur de l'application")
st.sidebar.write("[Lien vers le GitHub du créateur](https://github.com/mjacoupy)")
st.sidebar.markdown("Cette application est basée sur un modèle de Machine Learning !")
st.sidebar.markdown("Elle utilise un modèle de classification par Random Forest entrainé avec le légendaire jeu de données 'Titanic Survivor' de Kaggle.") 
st.sidebar.markdown("Ce modèle a un taux de bonne prédiction de 82,5% lorsqu'il s'agissait de prédire si une personne aurait réussi à monter sur un canot de sauvetage ou se serait noyé dans l'océan.")
st.sidebar.markdown("J'espère que cela vous plaira et n'oubliez pas : **Les femmes et les enfants d'abord !**")

# #######################################################################################################################

# 1. Insert image
st.image(img_titanic)

# #######################################################################################################################

# 2. Affiche le titre
st.title("Survivrez-vous au Titanic ?")

# #######################################################################################################################

# 3. Entrer le nom
name = st.text_input("Quel est votre nom ?")
st.markdown("""---""")

# #######################################################################################################################

# BONUS

# Bonus 1. Drag and drop photo
data = st.file_uploader("Ajouter une photo du passager", type=["png", "jpg", "jpeg"])


if data:
    image = Image.open(data)
    pil_image = Image.open(data).convert('RGB')
    open_cv_image = np.array(pil_image)
    image = cv2.cvtColor(open_cv_image[:, :, ::-1], cv2.COLOR_BGR2RGB)
    
    export_path = os.path.join(os.path.abspath(os.getcwd()))
    out_file = export_path + str(name) + ".png"
    cv2.imwrite(out_file, image)

# Bonus 2. Affiche photo en miniature
    scale_percent = 10
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    col1, col2 = st.columns([4, 6])
    with col1:
        st.image(resized)

# Bonus 3. Estime age et genre
    analyse = DeepFace.analyze(out_file, actions=['age', 'gender'])
    age = analyse['age']
    genre = analyse['gender']
    with col2:
        st.markdown("Age estimé : **"+str(age)+" ans**")
        if genre == 'Man':
            g = "Homme"
            s = 1
        elif genre == 'Woman':
            g = 'Femme'
            s = 0
        st.markdown("Genre estimé : **"+g+"**")
st.markdown("""---""")
# #######################################################################################################################

# 4. Choisir le genre 
if genre is None:
    col1, col2, col3 = st.columns(3)
        
    with col1:
        st.image(img_homme)
        male = st.checkbox("Homme")
        if male:
            s = 1
        
    with col2:
        st.image(img_femme)
        female = st.checkbox("Femme")
        if female:
            s = 0
            
    with col3:
        st.image(img_autre)
        other = st.checkbox("Non binaire")
        if other:
            s = 0.5       
    st.markdown("""---""")

# #######################################################################################################################

# 5. Selectionner l'age 
if age is None:
    age = st.number_input("Quel est votre age ?", step=1, min_value=1, max_value=120, value=20)
    st.markdown("""---""")

# #######################################################################################################################

# 6. Selectionner la classe
Class = st.selectbox("Dans quelle classe étiez-vous lors de votre dernier voyage ?", ['Affaire', 'Business', 'Economique'], index=0)
c = 0
if Class == "Affaire":
    c = 1
elif Class == "Business":
    c = 2
elif Class == "Economique":
    c = 3
st.markdown("""---""")

# #######################################################################################################################

# 7.1 Etes vous mariés
celib = st.radio("Etes-vous célibataire ?", ['Oui', 'Non'])
if celib == "Oui":
    m = 0
else:
    m = 1    
st.markdown("""---""")

# #######################################################################################################################

# 8. Voyagez vous avec vos frères et soeur ?
fs =  st.number_input("Combien de vos frères et/ou soeurs voyagent avec vous d'habitude ?", step=1, min_value=0)
st.markdown("""---""")

# #######################################################################################################################

# 9. Est ce que vous voyager avec vos enfants ? Si oui combien ?
enfants = st.radio("Avez-vous des enfants ?", ["Non", "Oui"], index=0)
if enfants == 'Oui':
    nb_enfants = st.number_input("Combien voyagent avec vous ?", step=1, min_value=0) 
else:
    nb_enfants = 0
st.markdown("""---""")    
  
# #######################################################################################################################

# 10. D'ou etes vous partis ? 
Port = st.selectbox("D'où souhaitez-vous partir ?", ["France", "Nord du Royaume-Uni", "Sud du Royaume-Uni"])
if Port == "France":
    p = 1
elif Port == "Nord du Royaume-Uni":
    p = 2
elif Port == "Sud du Royaume-Uni":
    p = 0
st.markdown("""---""")

# #######################################################################################################################

# 11. Generalement est ce que vous achetez vos billets en avance ou au dernier moment ?
Quand = st.radio("Quand achetez-vous généralement vos billets ?", ["Dés que possible", "Quelques semaines avant le depart", "Au dernier moment"])
if Quand == "Dés que possible":
    f = X_train.loc[X_train['Pclass'] == c]['Fare'].quantile([0.25])
elif Quand == "Quelques semaines avant le depart":
    f = X_train.loc[X_train['Pclass'] == c]['Fare'].quantile([0.5])
elif Quand == "Au dernier moment":
    f = X_train.loc[X_train['Pclass'] == c]['Fare'].quantile([0.75])
st.markdown("""---""")  
    
# #######################################################################################################################

# 12. Creation Dataframe 
button = st.button("Prédiction")
if name and button:
    try:
        dataframe = pd.DataFrame(np.array([[c, s, age, (m+fs), nb_enfants, f, p]]), columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])    
        pred = random_forest.predict(dataframe)[0]
        
        if pred == 0:
            st.error("Désolé, "+name+", vous n'auriez probablement pas survécu au naufrage du titanic")
            st.image(img_jack)
        elif pred == 1:
            st.success("Vous avez de la chance "+name+", vous auriez probablement survecu au naufrage du titanic")
            st.image(img_rose)
            st.balloons()
    except Exception as e:
        if s is None:
            st.warning("Veuillez sélectionner un genre")
elif button:
    st.warning("Veuillez renseigner votre nom")  
            
    
# #######################################################################################################################

#                                          # === END OF FILE === #

# #######################################################################################################################

    
    
    
    
    