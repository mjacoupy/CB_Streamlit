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

sns.set_theme()
# #######################################################################################################################
#                                              # === USER INPUT === #
# #######################################################################################################################

# Sexe pour le plot 1 ("Femme" ou "Homme")
Sex = "Homme"
# Classe pour le plot 2 (1, 2 ou 3)
# Port pour le plot 2 ("Cherbourg; "Queenstown; "SSouthampton)
Port = "Cherbourg"
Class = 1
# Passenger pour la prediction (de 1 à 332)
Passager = 331

# #######################################################################################################################
#                                              # === CODE === #
# #######################################################################################################################

# Data Prep

train_df = pd.read_csv('input/train.csv')
test_df = pd.read_csv('input/test.csv')


train_df = train_df.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
train_df['Sex'] = train_df['Sex'].apply(lambda x:1 if x=='male' else 0)
test_df['Sex'] = test_df['Sex'].apply(lambda x:1 if x=='male' else 0)
train_df['Embarked'] = test_df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
test_df['Embarked'] = test_df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


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
sub_df = train_df[train_df['Sex']==s]


g = sns.histplot(data=sub_df, 
                 x='Age',
                 color=c,
                 bins=20)


title = "Répartition des décés chez les "+Sex+"s"
g.set_title(title)




# Plot 2 : Deces par class et par port
if Port == "Cherbourg":
    p = 1
elif Port == "Queenstown":
    p = 2
elif Port == "Southampton":
    p = 0

sub_df = train_df[train_df['Embarked']==p]
sub_df2 = sub_df[sub_df['Pclass']==Class]

h = sns.barplot(data=sub_df2, 
                x='Sex',
                y='Survived')
title2 = "Survie pour les passagers de la class "+str(Class)+" étant parti de "+Port
h.set_title(title2)


# Data Prep
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()

random_forest = RandomForestClassifier(n_estimators=10)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


prediction = Y_pred[Passager-1]
if prediction == 1:
    print('Survived')
    print(test_df.iloc[[Passager-1]])
elif prediction == 0:
    print("Not survived")
    print(test_df.iloc[[Passager-1]])