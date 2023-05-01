#!/usr/bin/env python
# coding: utf-8

import pandas as pd

def pre_processing(file_path):
    '''
    Entréé : chemin du fichier a prétraiter
    Sortie : fichier prétraité
    '''
    # Lecture du fichier à pré-traiter
    df = pd.read_csv(file_path)
    
    # Sauvegarde de la colonne PassengerId
    #df_ids = df['PassengerId']

    # Age
    trainData = pd.read_csv('.//Data//train.csv')
    df['Age'] = df['Age'].fillna(trainData['Age'].median())
    df['Age'] = (df['Age'] - df['Age'].mean()) / df['Age'].std()

    # Sex
    df.loc[df.Sex == "male", 'Sex'] = '0'
    df.loc[df.Sex == "female", 'Sex'] = '1'

    # Embarked
    df.loc[df.Embarked == "C", 'Embarked'] = '1'
    df.loc[df.Embarked == "S", 'Embarked'] = '2'
    df.loc[df.Embarked.isna(), 'Embarked'] = '2'
    df.loc[df.Embarked == "Q", 'Embarked'] = '3'

    # Fare
    df['Fare'] = df['Fare'].fillna(trainData['Fare'].median())
    df['Fare'] = (df['Fare'] - df['Fare'].mean()) / df['Fare'].std()

    # retire Features moins importants
    features_drop = ['Name', 'Ticket', 'Cabin']
    df = df.drop(features_drop, axis=1)
    
    return df

if __name__ == '__main__' :
    
    # Pre-processing des donnée d'entrainement
    train = pre_processing('.//Data//train.csv')
    # Exportation des données d'entrainement après pre-processing, en CSV dans le répertoire courant
    train.to_csv('.//Data//train_preprocessed.csv', index=False)

    # Pre-processing des données test
    test = pre_processing('.//Data//test.csv')
    # Exportation des données test après pre-processing, en CSV dans le répertoire courant
    test.to_csv('.//Data//test_preprocessed.csv', index=False)

    # (BONUS)Pre-processing des données des membres du projet
    bonus = pre_processing('.//Data//bonus_data.csv')
    bonus.to_csv('.//Data//bonus_data_preprocessed.csv')



