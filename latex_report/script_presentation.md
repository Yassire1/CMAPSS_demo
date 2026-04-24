# Script de présentation — Prédiction Data-Driven de la RUL (CMAPSS)

---

## Slide 1 : Page de garde
Bonjour, je présente aujourd'hui mon projet de fin de module sur la prédiction de la vie utile résiduelle, appliquée au jeu de données CMAPSS de la NASA.

---

## Slide 2 : Plan
Voici le plan : d'abord l'introduction, ensuite les données et le prétraitement, puis les deux modèles XGBoost et LSTM, leur comparaison, la démo Streamlit, et enfin la conclusion.

---

## Slide 3 : Contexte et motivation
La maintenance prédictive permet d'anticiper les pannes avant qu'elles n'arrivent. L'objectif est de prédire la RUL, c'est-à-dire le temps restant avant la défaillance d'un composant. Cela évite les arrêts imprévus et réduit les coûts.

---

## Slide 4 : Objectifs du projet
Nous avons cinq objectifs : explorer le jeu de données CMAPSS, prétraiter les données, implémenter XGBoost et LSTM, évaluer les résultats, et enfin démontrer tout ça avec une application Streamlit.

---

## Slide 5 : CMAPSS — Description
Le jeu de données CMAPSS vient de la NASA. Il contient 100 moteurs turbofan simulés. Chaque observation a 26 colonnes. Nous avons trois fichiers : train avec les cycles jusqu'à défaillance, test avec des cycles partiels, et le fichier RUL qui donne la vérité terrain.

---

## Slide 6 : Analyse exploratoire — Capteurs
Voici les diagrammes en boîte des 21 capteurs. On voit que certains capteurs ne varient presque pas, donc ils ne sont pas utiles pour la prédiction.

---

## Slide 7 : Sélection des capteurs pertinents
Nous avons écarté 7 capteurs constants ou quasi-constants. Il reste 14 capteurs actifs. Les données sont normalisées entre moins un et un, moteur par moteur.

---

## Slide 8 : Similarité train / test
Cette figure compare les distributions entre les données d'entraînement et de test. Les distributions sont similaires, donc le modèle devrait bien généraliser.

---

## Slide 9 : Modélisation de la dégradation
Nous utilisons deux modèles de dégradation. Le linéaire fait diminuer la RUL de un à chaque cycle. Le linéaire par morceaux plafonne la RUL à 125 cycles au début, puis elle descend linéairement. Cette deuxième approche donne de meilleurs résultats.

---

## Slide 10 : Fenêtrage temporel
Pour le modèle LSTM, nous créons des fenêtres glissantes de 30 cycles avec un chevauchement maximal. Chaque entrée a trois dimensions : le nombre de fenêtres, 30 cycles d'historique, et 14 capteurs. Pour le test, on fait la moyenne des prédictions sur les dernières fenêtres de chaque moteur.

---

## Slide 11 : Métriques — RMSE et S-score
Nous utilisons deux métriques. Le RMSE mesure l'erreur quadratique moyenne entre la RUL réelle et la RUL prédite. Le S-score est asymétrique : il pénalise beaucoup plus les surestimations, car prédire une vie trop longue est plus dangereux qu'une maintenance trop précoce.

---

## Slide 12 : Principe de XGBoost
XGBoost signifie eXtreme Gradient Boosting. C'est un ensemble d'arbres de décision construits les uns après les autres. Chaque arbre corrige les erreurs du précédent. Pour XGBoost, nous devons aplatir les fenêtres en vecteurs. La fonction objectif combine la perte de prédiction et une régularisation.

---

## Slide 13 : Recherche d'hyperparamètres
Nous avons fait une validation croisée à dix plis sur une grille d'hyperparamètres. Les meilleurs paramètres trouvés sont une profondeur maximale de cinq et un taux d'apprentissage de zéro virgule un, avec cent rounds.

---

## Slide 14 : Résultats XGBoost
Le modèle XGBoost atteint un RMSE de 19 virgule 07 cycles et un S-score de mille vingt-quatre. Les prédictions sont correctes dans l'ensemble, mais le S-score élevé montre qu'il y a des surestimations coûteuses sur certains moteurs.

---

## Slide 15 : Importance des variables
Voici l'importance des variables pour XGBoost. Les capteurs de température et de pression sont les plus informatifs. Par contre, XGBoost ne capture pas explicitement la dimension temporelle.

---

## Slide 16 : Pourquoi les LSTM ?
Les réseaux LSTM sont conçus pour les données séquentielles. Ils ont une mémoire interne et trois portes : oubli, entrée, sortie. Cela leur permet de capturer les dépendances long terme, ce qui est parfait pour la dégradation des moteurs sur des centaines de cycles.

---

## Slide 17 : Architecture du réseau LSTM
Notre réseau a trois couches LSTM empilées, suivies de deux couches denses. L'entrée a 30 cycles sur 14 capteurs. Le modèle compte environ 223 mille paramètres.

---

## Slide 18 : Protocole d'entraînement
Nous utilisons l'optimiseur Adam avec un taux d'apprentissage qui passe de 0.001 à 0.0001 après cinq époques. L'entraînement dure seulement dix époques pour éviter le surapprentissage. On utilise 80% des données pour l'entraînement et 20% pour la validation.

---

## Slide 19 : Résultats LSTM
Le modèle LSTM atteint un RMSE de 15 virgule 17 cycles et un S-score de 448. C'est nettement mieux que XGBoost. Le modèle suit bien la RUL réelle et fait moins de surestimations.

---

## Slide 20 : Tableau comparatif
En résumé, le LSTM bat XGBoost sur toute la ligne. Le RMSE est 20% plus bas, le S-score est divisé par plus de deux. XGBoost est plus rapide à entraîner et plus interprétable, mais le LSTM capture mieux le temps.

---

## Slide 21 : Discussion
Le LSTM offre une meilleure précision et une modélisation explicite du temps. Ses limites sont une complexité accrue, un temps d'entraînement plus long, et une faible interprétabilité. Les poids du modèle sont archivés pour garantir la reproductibilité.

---

## Slide 22 : Démonstration Streamlit
Nous avons aussi développé une application Streamlit interactive. Elle permet de sélectionner un moteur, visualiser ses capteurs, obtenir une prédiction en temps réel, et comparer la RUL réelle avec la RUL prédite.

---

## Slide 23 : Conclusion
En conclusion, nous avons construit un pipeline complet de prétraitement, comparé XGBoost et LSTM, montré la supériorité du LSTM, et proposé une application de démonstration. Les perspectives incluent l'extension aux autres sous-ensembles, l'ajout de mécanismes d'attention, et le déploiement embarqué.

---

## Slide 24 : Merci
Merci de votre attention. Je suis disponible pour vos questions.
