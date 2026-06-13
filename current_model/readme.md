1 classe normale (ex : 0), plusieurs types de défauts (les autres chiffres)
ou **une dizaine de négatifs (pas des défauts) et un ou plusieurs défauts (lettres A, B)**

étape suivante : développer un classifier capable de détecter positifs et laisser passer négatifs pour les modèles de détection. Data augmentation classique suffit pas. 
**Objectif** : évaluer ou est la limite sur le nombre de positifs. 

GPU : plmlatex

Nous allons continuer le projet sur MNIST. Cependant, nous allons utiliser plusieurs lettres qui représenteront les différents types de défauts et les chiffres représenteront les différents types de normaux. Il faudra donc leur ajouter des labels car il faut reconnaitre le type de défaut. Qu'y a t-il a changé (ne code pas mais dis moi ce qu'il se passe) ?
De plus, voici un schema plus clair du but de mon modèle : Dataset réduit - (mon modèle DDMM Lora) -> Dataset augmenté - (entrainement détecteur) -> test du détecteur sur le dataset : combien de positifs faut il au minimum pour détecter un grand % de défaut.