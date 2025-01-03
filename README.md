# Document Detector (doc_detector_pl)

**Repository** : [https://github.com/AntoineBendafiSchulmann/doc_detector_pl](https://github.com/AntoineBendafiSchulmann/doc_detector_pl)

## Description du Projet

Le projet **Document Detector** utilise **PyTorch Lightning** pour entraîner un modèle de détection basé sur l'architecture **UNet** qui est une structure de réseaux de neurones conçu pour la segmentation d'images, c'est à dire reconnaitre et localiser des éléments spécifiques dans une image. Il permet de détecter, segmenter et recadrer automatiquement des documents dans des images. Le projet est organisé pour un environnement d'entraînement et d'évaluation flexible, en exploitant TensorBoard pour visualiser les courbes d'entraînement et OpenCV pour le traitement des masques.

---

## Structure du Projet

![Exemple d'image](assets/structure_files.png)



- **`data/`** : Données utilisées pour l'entraînement et les tests.
  - `images/` : Images d'entraînement.
  - `masks/` : Masques d'entraînement correspondants.
  - `debug/` : Contient les images de debug, où les masques générés sont superposés aux images originales. Ces fichiers permettent de vérifier visuellement la qualité et la pertinence des masques.
  - `test_images/` : Images de test pour vérifier les performances du modèle.

- **`models/`** : Fichiers des modèles sauvegardés :
  - `.pth` : Modèle PyTorch (format entraînable).
  - `.onnx` : Modèle exporté pour l'inférence.

- **`outputs/`** : Comparaisons générées entre les images originales, les masques attendus, les prédictions, et les recadrages.

- **`logs/`** : Fichiers de logs générés par PyTorch Lightning pour le suivi via TensorBoard.

- **Scripts principaux** :
  - `generate_masks.py` : Génère des masques à partir des images d'entraînement.
  - `train_model.py` : Lance l'entraînement du modèle.
  - `test_model.py` : Effectue une prédiction sur les images de test, génère les comparaisons, et recadre les images.

---

## Installation et Configuration

1. **Cloner le dépôt** :
   ```bash
   git clone https://github.com/AntoineBendafiSchulmann/doc_detector_pl
   cd doc_detector_pl
   ```

2. **Installer les dépendances** :
   ```bash
    pip install -r requirements.txt
   ```

    #### Attention -> à lire !!!
   j'ai généré un fichier ```requirements.txt``` contenant la liste des dépendances installées sur mon environnement virtuel python (il se peut qu'il y ait des trucs pertinent pour moi dedans  genre certaines bibliothèques comme ```pytorch-triton-rocm``` ou ```jax``` pour les gpu amd désolé)

   Ceci est la commande pour générer le fichier directement à partir de son environnement virtuel  :

    ```bash
   pip freeze > requirements.txt
    ```



## Utilisation

###  Génération des Masques
Pour préparer les données d'entraînement, des masques correspondant aux documents présents dans les images sont générés à l'aide du script ```generate_masks.py```. Les masques servent de cibles pour l'entraînement du modèle

```bash
python generate_masks.py
```

####  Fonctionnement
Le script utilise OpenCV pour détecter automatiquement les documents dans les images. Voici les étapes principales :

- Floutage des Images : Utilise un flou gaussien pour réduire les bruits.
- Seuil adaptatif : Applique une méthode de seuil adaptatif pour séparer le fond des zones importantes.
- Contours : Identifie les contours dans l'image.
- Filtrage : Retient uniquement les contours ayant :
- Quatre côtés (approximé comme un quadrilatère).
- Une aire minimale (par défaut, > 1000 pixels).
- Masque Final : Dessine le quadrilatère détecté.

##### Les masques générés sont sauvegardés dans le dossier ```data/masks/```

#### Débogage Visuel

En cas de besoin, un répertoire ```debug/``` est utilisé pour sauvegarder des images où les masques générés sont superposés aux images originales. Cela permet de vérifier visuellement que les contours détectés et les masques générés sont conformes aux attentes.

- Les fichiers dans ```debug/``` aident à repérer rapidement les images problématiques.


### Entraîner le Modèle
Dans le terminal principal, exécutez la commande suivante pour lancer l'entraînement :

```bash
python train_model.py
```

Ouvrez un deuxième terminal, placez-vous dans le dossier ```doc_detector_pl``` et exécutez:

```bash
tensorboard --logdir logs/ --bind_all
```
ouvrez l'url pour accéder à TensorBoard

### Résultats générés :
- Les courbes d'entraînement sont visibles dans TensorBoard.
- Le modèle entraîné est sauvegardé dans le dossier ```models/```.


### Tester le Modèle
Testez le modèle sur une image de test et générez des comparaisons visuelles :

```bash
python test_model.py
```

####  Sorties générées :
- Les comparaisons (image originale, masque attendu, masque prédit, image recadrée) sont sauvegardées dans ```outputs/```.


### Explication des Courbes TensorBoard et Analyse des Risques

Les courbes affichées dans TensorBoard sont des outils essentiels pour suivre et comprendre l'entraînement du modèle. Elles permettent d'identifier les progrès réalisés mais aussi de détecter des problèmes comme l'overfitting.


#### 1. Progression des Époques (```epoch```)

- Description : Cette courbe indique le nombre d'époques complétées au fur et à mesure de l'entraînement.
##### - Interprétation :
- Elle sert simplement à visualiser la progression. Une augmentation linéaire est attendue et normale.

![Exemple d'image](assets/epoch.png)

#### 2. Perte Moyenne par Époque (```train_loss_epoch```)

- Description : Cette courbe montre comment la "perte", une mesure de l'erreur entre la prédiction du modèle et les résultats attendus, évolue à chaque époque (un passage complet de toutes les images du jeu d'entraînement dans le modèle).
##### - Interprétation :
- Une courbe qui diminue régulièrement indique que le modèle apprend bien.
- Si la perte stagne ou augmente après une phase de diminution, cela peut signaler que le modèle atteint sa capacité maximale ou commence à mémoriser les données (overfitting).

![Exemple d'image](assets/train_loss_epoch.png)


#### 3. Perte à Chaque Étape de Traitement (```train_loss_step```)

- Description : Cette courbe montre les variations de la perte après chaque groupe d'images (mini-lot ou "batch") traité par le modèle.
##### - Interprétation :
- De légères fluctuations sont normales et reflètent la diversité des données.
- Une tendance globale à la baisse est un bon signe. Cependant, si les variations sont trop importantes ou ne diminuent pas, cela peut indiquer que l'apprentissage est instable.

![Exemple d'image](assets/train_loss_step.png)

####  4. Perte sur les Données de Validation (```val_loss```)
- Description :
Cette courbe représente l'évolution de la perte (ou erreur) calculée sur les données de validation après chaque époque. Contrairement à la perte d'entraînement, la perte de validation mesure la capacité du modèle à généraliser sur des données qu'il n'a jamais vues auparavant.

#####  - Interprétation :
- Tendance à la baisse : Si la courbe diminue régulièrement, cela signifie que le modèle apprend à bien généraliser sur de nouvelles données.
- Augmentation ou stagnation : Si la courbe stagne ou augmente alors que la perte d'entraînement continue de diminuer, cela peut indiquer un début d'overfitting.
- Oscillations : Une légère variation est normale. Cependant, des fluctuations importantes peuvent refléter une instabilité dans l'apprentissage ou un jeu de validation trop petit.(dans mon cas actuel il se pouurait qu'on soit pas bon xd)

![Exemple d'image](assets/val-loss.png)


### Le Risque d'Overfitting : Comprendre et Éviter

##### Qu'est-ce que l'overfitting ?
L'overfitting se produit lorsque le modèle apprend trop bien sur les données d'entraînement, au point de mémoriser des détails spécifiques qui ne se généralisent pas à de nouvelles données. Cela se traduit par une bonne performance sur les données d'entraînement, mais de mauvaises performances sur des données jamais vues auparavant. Voici quelques techniques pour éviter ce problème

#####  Signes d'overfitting dans TensorBoard :
- Augmentation de la ```val_loss``` : Si la courbe de validation (```val_loss```) commence à augmenter ou stagner alors que la perte d'entraînement (```train_loss```) continue de diminuer, cela indique que le modèle mémorise les données d'entraînement sans généraliser correctement.
- Écart significatif entre ```train_loss``` et ```val_loss``` : Une différence importante entre la perte d'entraînement (très faible) et la perte de validation (plus élevée) reflète un sur-apprentissage sur les données d'entraînement.
- Fluctuations importantes de ```val_loss``` : De grandes oscillations peuvent indiquer que le modèle est instable ou que les données de validation ne sont pas représentatives.

#####  Conseils pour éviter l'overfitting :

#### 1.Améliorer les données d'entraînement :
- Augmenter le volume des données : Ajouter davantage d'exemples, si possible, pour couvrir une plus large gamme de variations (angles, résolutions, types de documents).
- Augmentation des données (Data Augmentation) : Utiliser des transformations comme des rotations, des inversions, des changements de luminosité ou des zooms. Cela peut être implémenté via des outils comme torchvision.transforms.

#### 2.Cross-validation :
- Diviser les données en plusieurs groupes et entraîner le modèle sur différents sous-ensembles de données. Cela aide à vérifier la capacité de généralisation du modèle et réduit la dépendance à une seule division "données d'entraînement / validation".

#### 3. Arrêt anticipé (Early Stopping) :
- Configurer un mécanisme dans PyTorch Lightning pour arrêter l'entraînement lorsque la perte de validation (```val_loss```) n'améliore plus après un certain nombre d'époch. Cela empêche le modèle d'apprendre des détails inutiles des données d'entraînement.
  
#### 4.Ajouter un Dropout :
- Utiliser des couches de Dropout dans le modèle **U-Net** pour désactiver aléatoirement une fraction des neurones pendant l'entraînement. Cela force le modèle à s'appuyer sur différentes combinaisons de caractéristiques et améliore la généralisation.

#### 5.Régularisation L2 :
- Ajouter une régularisation L2 (via l'option ```weight_decay``` dans l'optimiseur Adam). Cela limite la croissance excessive des poids du modèle et empêche la mémorisation.

#### 6.Simplifier le modèle si nécessaire :
- Si le modèle est trop complexe pour la taille des données disponibles (par exemple, trop de couches ou de paramètres), réduire sa taille peut améliorer la généralisation.

#### 7.Créer des ensembles de validation équilibrés :
- S'assurer que les données de validation reflètent la diversité des données réelles. Cela permet d'avoir une meilleure idée des performances du modèle sur des cas non vus.


### Fonctionnalité de Recadrage Automatique

Le script ```test_model``` permet d'effectuer une prédiction sur une image de test et de recadrer automatiquement le document détecté en fonction du masque prédit par le modèle UNet. Cette étape simule comment le modèle peut être utilisé dans un pipeline complet quand le modèle sera exporté sur tensorflow js pour être utilisable sur un navigateur, allant de la détection à l'extraction ciblée d'un document.


#### Étapes de Fonctionnement :
1. Chargement de l'image et du masque attendu :

- L'image de test est lue et préparée pour être passée dans le modèle.
- Un masque attendu peut également être utilisé pour comparaison. 

2. Prédiction du masque :
- Le modèle génère un masque binaire prédisant les zones correspondant au document.

3. Redimensionnement du masque :

- Le masque prédit est redimensionné pour correspondre aux dimensions originales de l'image.

4. Détection et recadrage :

- À l'aide des coordonnées du masque prédit, la zone correspondant au document est recadrée automatiquement.

5. Visualisation et sauvegarde :

##### - Quatre images sont générées :
- L'image originale
- Le masque attendu
- Le masque prédit
- L'image recadrée

#### - Ces résultats sont sauvegardés dans le dossier ```outputs/```.
Exemple de Commande :
Pour tester cette fonctionnalité, exécutez simplement la commande suivante :

```bash
python test_model.py
```

Voici la sortie générée par ce script :

![Exemple d'image](assets/result.png)

(je sais il reste du travail au vu du résultat mais vous avez l'idée)

- Image originale : Affiche l'image d'entrée brute.
- Masque attendu : Masque utilisé pour l'entraînement.
- Masque prédit : Généré automatiquement par le modèle.
- Image recadrée : Image finale extraite en fonction des prédictions

### Intégration en Production

Le modèle entraîné peut être exporté dans différents formats comme ONNX ou TensorFlow.js, permettant son utilisation dans des systèmes backend ou frontend.
- **ONNX** : Permet l'intégration avec des frameworks comme ONNX Runtime pour des prédictions rapides.
- **TensorFlow.js** : Rendra le modèle directement utilisable dans les navigateurs.

#### Déploiement via une API Flask

Le modèle peut également être déployé via une **API Flask** pour permettre une interaction en temps réel avec des systèmes externes. Voici les étapes principales d'une intégration avec Flask :

#### **Création de l'API** : L'API reçoit une image en entrée (par exemple sous forme de fichier envoyé dans une requête POST), l'analyse avec le modèle U-Net, et retourne :

- Le masque prédit.
- L'image recadrée.
- Les métriques associées.
  
