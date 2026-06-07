# NanoOracle — contexte projet (hackathon D4GEN)

## Quoi

Outil ML pour prédire les propriétés physicochimiques des nanoparticules lipidiques (LNP)
avant l'expérience, afin de dépasser un verrou R&D : la formulation LNP est empirique,
prend 6-12 mois par particule, et la majorité des essais échouent.

Livrables hackathon : prototype/mvp fonctionnel + pitch de 5 min.
Critères de notation : faisabilité, innovation, maturité/rapidité de mise en œuvre,
potentiel, pertinence.

## L'insight central (le pivot du projet)

On a découvert un résultat contre-intuitif : agréger les données de plusieurs labos
DÉGRADE le modèle (R² s'effondre) par rapport à un modèle entraîné sur un seul labo.
Cause hypothétique : biais de protocole spécifiques à chaque labo (instrument,
méthode de caractérisation, conditions de mixing, calibration, opérateur) non capturés.
Le modèle voit cette variance inter-labo comme du bruit irréductible.

=> Le vrai verrou n'est pas la QUANTITÉ de données mais leur NON-COMPARABILITÉ inter-labo.

## La solution proposée

1. Un pipeline PAR LABO qui entraîne un modèle local performant (valeur immédiate,
   R² élevé, pas besoin d'attendre une masse critique).
2. Ce pipeline standardise et loggue les MÉTADONNÉES DE PROTOCOLE que personne ne capture.
3. Une fois ces métadonnées explicites, un modèle hiérarchique (partial pooling /
   effets aléatoires par labo) peut séparer le signal physico-chimique partagé du biais
   propre à chaque labo => un modèle général devient enfin possible.

## État réel (à être honnête, ne pas survendre)

- Modèle : XGBoost. Choix assumé (bon outil pour petit dataset tabulaire, interprétable
  via SHAP). NE PAS rebrander en LLM/IA générative.
- Données : petit dataset issu de la thèse d'un membre de l'équipe.
- Le résultat R² (N=1 > N>1) est OBSERVÉ et réel.
- Le mécanisme (capturer le contexte labo récupère le R²) est une HYPOTHÈSE non encore
  démontrée — on n'a pas les métadonnées de protocole dans le dataset actuel.

## Déploiement

Dockerfile qui marche en local et ./deploy.sh pour deploy l'image docker sur aws automatiquement.

## Répartition équipe

Data scientists : le modèle (XGBoost, ablation, learning curve, hiérarchique).
Moi : le proto/front.

Garde-fous : ne pas survendre le modèle global comme "déjà marche" (ta donnée actuelle
le contredit). Question-piège à anticiper : "si chaque labo a son bon modèle local,
qu'apporte le global ?" → généralisation hors du domaine exploré par un labo, via partial
pooling corrigé du biais.
