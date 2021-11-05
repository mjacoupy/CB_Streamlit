#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 15:31:10 2021

@author: maximejacoupy
"""
# #######################################################################################################################
#                                              # === LIBRAIRIES === #
# #######################################################################################################################
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

import streamlit as st

sns.set_theme()

# #######################################################################################################################
#                                              # === STREAMLIT === #
# #######################################################################################################################
st.title("Exemple Titanic - Streamlit")
st.markdown("""---""")

analysis = st.sidebar.selectbox('', ['Histogramme', 'Barplot', 'Prediction'])

# #######################################################################################################################
#                                              # === USER INPUT === #
# #######################################################################################################################


# Sexe pour le plot 1 ("Femme" ou "Homme")
Sex = "Homme"
# Classe pour le plot 2 (1, 2 ou 3)
# Port pour le plot 2 ("Cherbourg; "Queenstown; "Southampton)
Port = "Cherbourg"
Class = 1
# Passenger pour la prediction (de 1 à 332)
Passager = 331

# #######################################################################################################################
#                                              # === CODE === #
# #######################################################################################################################

if analysis == 'Histogramme':

    Sex = st.radio("Sexe", ['Homme', 'Femme'])

    # Data Prep
    train_df = pd.read_csv('input/train.csv')
    test_df = pd.read_csv('input/test.csv')

    train_df = train_df.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1)
    test_df = test_df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    train_df['Sex'] = train_df['Sex'].apply(lambda x: 1 if x == 'male' else 0)
    test_df['Sex'] = test_df['Sex'].apply(lambda x: 1 if x == 'male' else 0)
    train_df['Embarked'] = test_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    test_df['Embarked'] = test_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
    train_df = train_df.dropna()
    test_df = test_df.dropna()


    # Plot 1 : Survived by Age
    if Sex == "Femme":
        s = 0
        c = "orange"
    elif Sex == "Homme":
        s = 1
        c = "darkblue"
    sub_df = train_df[train_df['Sex'] == s]

    g = plt.figure()
    sns.histplot(data=sub_df,
        x='Age',
        color=c,
        bins=20).set_title("Répartition des décés chez les "+Sex+"s")

    st.pyplot(g)

if analysis == 'Barplot':
    col1, col2, col3 = st.columns(3)
    with col1:
        Port = st.selectbox("Ville", ['Cherbourg', 'Queenstown', 'Southampton'])
    with col2:
        Class = st.slider('Classe', min_value=1, max_value=3)
    with col3:
        Button = st.button("Plot")

    if Port and Class and Button:

        # Data Prep
        train_df = pd.read_csv('input/train.csv')
        test_df = pd.read_csv('input/test.csv')

        train_df = train_df.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1)
        test_df = test_df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
        train_df['Sex'] = train_df['Sex'].apply(lambda x: 1 if x == 'male' else 0)
        test_df['Sex'] = test_df['Sex'].apply(lambda x: 1 if x == 'male' else 0)
        train_df['Embarked'] = test_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
        test_df['Embarked'] = test_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

        test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
        train_df = train_df.dropna()
        test_df = test_df.dropna()

        # Plot 2 : Deces par class et par port
        if Port == "Cherbourg":
            p = 1
        elif Port == "Queenstown":
            p = 2
        elif Port == "Southampton":
            p = 0

        sub_df = train_df[train_df['Embarked'] == p]
        sub_df2 = sub_df[sub_df['Pclass'] == Class]

        f = plt.figure()
        sns.barplot(data=sub_df2,
            x='Sex',
            y='Survived').set_title("Survie pour les passagers de la classe "+str(Class)+" étant parti de "+Port)
        st.pyplot(f)

if analysis == 'Prediction':
    Passager = st.number_input('Passager', min_value=0, max_value=332)
    Button = st.button('Predict')

    if Passager and Button:

        train_df = pd.read_csv('input/train.csv')
        test_df = pd.read_csv('input/test.csv')

        train_df = train_df.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1)
        test_df = test_df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
        train_df['Sex'] = train_df['Sex'].apply(lambda x: 1 if x == 'male' else 0)
        test_df['Sex'] = test_df['Sex'].apply(lambda x: 1 if x == 'male' else 0)
        train_df['Embarked'] = test_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
        test_df['Embarked'] = test_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

        test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
        train_df = train_df.dropna()
        test_df = test_df.dropna()

        # Data Prep
        X_train = train_df.drop("Survived", axis=1)
        Y_train = train_df["Survived"]
        X_test = test_df.drop("PassengerId", axis=1).copy()

        random_forest = RandomForestClassifier(n_estimators=10)
        random_forest.fit(X_train, Y_train)
        Y_pred = random_forest.predict(X_test)
        random_forest.score(X_train, Y_train)
        acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

        prediction = Y_pred[Passager-1]
        if prediction == 1:
            st.text('Survived')
            st.text(test_df.iloc[[Passager-1]])
        elif prediction == 0:
            st.text("Not survived")
            st.text(test_df.iloc[[Passager-1]])
