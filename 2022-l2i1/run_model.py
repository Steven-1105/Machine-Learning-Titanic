import pandas as pd
import numpy as np
# Importation du modèle de machine learning 'Random Forest Classifier' :
from sklearn.ensemble import RandomForestClassifier
# Importation de la fonction permettant de diviser notre dataframe :
from sklearn.model_selection import train_test_split
# Importation de la fonction permettant de mesurer la précision du modèles :
from sklearn.metrics import accuracy_score

if __name__ == '__main__' :
    # Importation des fichiers via Pandas :
    train = pd.read_csv('Data/train_preprocessed.csv')
    test = pd.read_csv('Data/test_preprocessed.csv')

    # Séparons notre fichier train en 4 fichiers différents : 
    # X_train : contenant les informations sur les passagers sans la colonne "Survived" sur lequel on entrainera les modèles
    # y_train : la colonne "Survived" correspondant à X_train sur laquelle on entrainera les modèles
    # X_test : contenant les informations sur les passagers sans la colonne "Survived" sur lequel on testera les modèles
    # y_test : la colonne "Survived" correspondant à X_train sur laquelle on testera les modèles

    y = train["Survived"]
    X = train.drop('Survived', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) 
    # test_size = 0.3 signifie que la taille de X/y_test correspond à 30% de train et donc que celle de X/y_train correspond à 70% de train

    # Afin de limiter les effets de l'aléatoire sur la comparaison du résultat, nous utilisons random.seed() qui conservera les mêmes valeurs aléatoires à chaque test des modèles :
    np.random.seed(42)


    # Random Forest Classifier :
    # les hyper-paramètres utilisés seront ceux qui auront été définis comme les meilleurs après multiples essais. (voir NOtebook Jupyter)
    # Dans un premier temps on test le modèle :

    # On entraine le modèle sur 70% du fichier 'train' :
    RFC = RandomForestClassifier(max_depth = 13, n_estimators = 160)
    RFC.fit(X_train, y_train)

    # On test le modèle sur les 30% restants : 
    X_test_prediction = RFC.predict(X_test)

    # accRFC est le taux de réussite du modèle sur les données d'entrainement : 
    accRFC = accuracy_score(y_test, X_test_prediction)

    # On écrit la précision dans un fichier texte que l'on créer dans le repertoire courant
    with open('Resultats/précision_du_modele.txt', 'w') as f:
        f.write("Précision du 'Random Forest Classifier' : {:.4f}".format(accRFC))
    
    # X_train devient train sans la colonne "Survived"
    # y_train devient la colonne "Survived" de train
    # X_test devient test
    # Nous ne possédons pas y_test, c'est Kaggle qui "fera le test de précision".

    # A présent, on utilise le modèle pour prédire les données de test et créer un fichier de soumission lisible par Kaggle :

    # Cette fois-ci, on entraine le modèle avec l'ensemble des données d'entrainement : 
    RFCSoumission = RandomForestClassifier(max_depth = 13, n_estimators = 160)
    RFCSoumission.fit(train.drop("Survived", axis=1), train["Survived"])
    # On fais tourner le modèle sur l'ensemble des données test :
    predictions = RFCSoumission.predict(test)
    # Création du fichier de soumission :
    soumission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': predictions})
    soumission.to_csv('Resultats/soumission.csv', index=False) # on créer le fichier soumission dans le répertoire courant




    # Bonus, on test le modèle sur les membres du projet pour déterminer le ou lesquels d'entre nous auraient survécus :
    bonus = pd.read_csv('Data/bonus_data_preprocessed.csv')
    bonus = bonus.drop(columns=['Unnamed: 0'])
    RFCBonus = RandomForestClassifier(max_depth = 13, n_estimators = 160)
    RFCBonus.fit(train.drop("Survived", axis=1), train["Survived"])

    bonus_pred = RFCSoumission.predict(bonus)
    correspondance = {1: "Mathieu Antonopoulos", 4: "Melissa Merabet", 3: "Timothé Miel", 2: "Hongxiang Lin"}

    result_df = pd.DataFrame({'Nom_Prenom': [correspondance.get(id, f"Passager {id}") for id in bonus['PassengerId']],
                             'Prediction': bonus_pred})
    result_df['Prediction'] = result_df['Prediction'].map({0: 'mort(e)', 1: 'vivant(e)'})
    # Créer un fichier texte avec les résultats dans le répertoire courant
    result_df.to_csv('Resultats/bonus_predictions.txt', index=False, sep='\t', header=False)





