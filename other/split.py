import os
import shutil
import random
from sklearn.model_selection import train_test_split

# Chemin vers votre dossier de dataset
chemin_du_dossier_dataset = './EuroSAT_data'

# Liste de toutes les classes dans le dossier de dataset
classes = os.listdir(chemin_du_dossier_dataset)

# Création des répertoires pour les ensembles de train et de test
chemin_entrainement = './data/train'
chemin_test = './data/test'

if not os.path.exists(chemin_entrainement):
    os.makedirs(chemin_entrainement)

if not os.path.exists(chemin_test):
    os.makedirs(chemin_test)

# Séparation des données en ensembles d'entraînement et de test (80 % train, 20 % test pour chaque classe)
for nom_classe in classes:
    print(nom_classe)
    images_classe = os.listdir(os.path.join(chemin_du_dossier_dataset, nom_classe))
    images_entrainement, images_test = train_test_split(images_classe, test_size=0.2, random_state=42)

    # Création des dossiers pour chaque classe dans les ensembles d'entraînement et de test
    chemin_classe_entrainement = os.path.join(chemin_entrainement, nom_classe)
    chemin_classe_test = os.path.join(chemin_test, nom_classe)

    if not os.path.exists(chemin_classe_entrainement):
        os.makedirs(chemin_classe_entrainement)

    if not os.path.exists(chemin_classe_test):
        os.makedirs(chemin_classe_test)

    # Copie des images dans les dossiers d'entraînement et de test respectifs
    for img in images_entrainement:
        shutil.copy(os.path.join(chemin_du_dossier_dataset, nom_classe, img), os.path.join(chemin_classe_entrainement, img))

    for img in images_test:
        shutil.copy(os.path.join(chemin_du_dossier_dataset, nom_classe, img), os.path.join(chemin_classe_test, img))