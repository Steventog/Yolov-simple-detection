# Système de Détection et Tracking d'Objets

## Description
Application Python de détection et tracking d'objets en temps réel utilisant YOLOv8 et l'algorithme SORT. Le système permet de détecter, tracker et compter des objets soit via une webcam en temps réel, soit à partir d'une vidéo préenregistrée.

## Fonctionnalités
- 🎥 Détection en temps réel via webcam
- 📹 Analyse de vidéos préenregistrées
- 🖥️ Interface graphique simple

## Prérequis
- Python 3.12
- GPU recommandé pour de meilleures performances

## Installation

1. Cloner le repository
```bash
git clone https://github.com/votre-username/Detection_objet.git
cd Detection_objet
```

2. Créer un environnement virtuel
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

3. Installer les dépendances
```bash
pip install -r requirements.txt
```

4. Télécharger le modèle YOLOv8
```bash
# Le modèle sera automatiquement téléchargé au premier lancement
```

## Utilisation

1. Lancer l'application
```bash
python main.py
```

2. Interface
- Cliquer sur "Détection en temps réel" pour utiliser la webcam
- Cliquer sur "Charger une vidéo" pour analyser une vidéo
- Utiliser les contrôles vidéo pour :
  * Arrêter la vidéo en cours
  * Charger une nouvelle vidéo


## Dépendances principales
- ultralytics==8.3.64
- opencv-python>=4.8.0
- numpy>=1.26.2
- scipy>=1.15.1
- filterpy>=1.4.5

## Contribution
Les contributions sont les bienvenues ! N'hésitez pas à :
1. Fork le projet
2. Créer une branche pour votre fonctionnalité
3. Commiter vos changements
4. Pousser vers la branche
5. Ouvrir une Pull Request

## Licence
Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

## Auteur
Steventog

## Remerciements
- Ultralytics pour YOLOv8
- Alex Bewley pour l'algorithme SORT
