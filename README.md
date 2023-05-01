# Machine-Learning-Titanic 
## Projet L2I1 (Université Paris Cité)
## Description

Ce projet a été réalisé en 2023 dans le cadre de notre 2eme année de licence informatique à l'Université de Paris cité.
Le projet vise à construire un modèle prédictif par Machine Learning, en utilisant les données relatives aux passagers du Titanic, afin de prédire qui aurait le plus de chances de survivre au naufrage. 
Le projet se base sur un ensemble de données, 'train.csv', contenant les informations sur les passagers tels que le nom, l'âge, le sex, la classe socio-économique, etc. On dispose également d’un fichier ‘test.csv’ permettant d’évaluer la précision de notre modèle.

## Fonctionnement 

Deux fichiers écris en python, permettent de tester le modèle. Un Notebook Jupyter permet de visualiser les étapes de conception du modèle et comprendre son fonctionnement.

## Contenu du projet

- Un répertoire 'Data_from_kaggle' contenant les documents ‘test.csv’ et ‘train.csv’ fournis par Kaggle.
- Un répertoire 'Documentation' contenant le guide d'exécution des fonctions python et un guide d'installation de l'environnement Conda afin de consulter le Notebook de façon intéractive.
- Un fichier ‘pre_processing.py’ : Ce document exécute une fonction python permettant d’appliquer la fonction de prétraitement aux sets données ‘test.csv’ et ‘train.csv’.
- Un fichier ‘run_model.py’ : Ce document est une fonction python permettant de faire tourner le modèle sur les sets de données déjà traités. Ainsi, ce code permettra de faire tourner le modèle sur les données d'entraînement afin de tester son efficacité. De plus, un fichier de soumission sera créé. Ce fichier ‘soumission.csv’ donnera les prédictions du modèle sur les données test. Le format du fichier est celui demandé par Kaggle et permet donc d’obtenir un score.
- Un fichier ‘Notebook.ipynb’ : Notebook Jupyter commentant toutes les étapes de pré-processing, entraînement et soumission. De plus ce Notebook retrace une partie des analyses et recherches réalisées au cours du projet afin d’assurer une meilleure interprétation des résultats.
- Un fichier‘Notebook.pdf’: version pdf de ‘Notebook.ipyb’.
- Un fichier csv 'bonus_data.csv'qui contient les informatations des membres du groupe sous un format compatible avec la fonction de prétraitement.


## Utilisation

Pour exécuter les programmes Pyhton, veuillez consulter le guide d'execution et suivre les instructions.
Pour consulter le Notebook sur Jypter, veuillez consulter le guide d'installation de l'environnement Conda et suivre les indications 

## Prérequis

- Un ordinateur avec un processeur compatible avec Python 3.10.
- Un espace de stockage suffisant pour les fichiers de données prétraitées et les
résultats de l'exécution du modèle. (512 MB RAM et 1GB de stockage minimum)

## Auteurs 

Hongxiang Lin, Mathieu Antonopoulos, Melissa Merabet, Timothé Miel
