# Analyse des corrélations entre météo et rendements d’actifs financiers

## Objectif

Ce projet a pour objectif d’explorer l’existence de corrélations potentielles entre certaines conditions météorologiques et les rendements journaliers de plusieurs actifs financiers. L’approche est volontairement exploratoire et ne vise pas à établir de relation causale, mais à identifier d’éventuels patterns statistiques suggérant que certains actifs puissent réagir différemment selon le contexte météorologique.

## Données utilisées

- Prix journaliers d’actifs financiers via Yahoo Finance :
  - S&P 500  
  - VIX  
  - Natural Gas  
  - Utilities (XLU)  
  - Corn futures (CORN)

- Données météorologiques journalières via Meteostat :
  - station de New York (phase initiale)
  - stations situées dans la Corn Belt américaine (phase ultérieure)

## Méthodologie

1. Collecte et nettoyage des données financières et météorologiques.  
2. Alignement des données météo sur les jours d’ouverture des marchés.  
3. Construction d’indicateurs d’événements météo :
   - neige  
   - pluie intense  
   - vent fort  
   - forte couverture nuageuse  
   - humidité élevée  
   - conditions favorables (“beau temps”)  
   - jours sans événement (“normaux”)

4. Analyse descriptive des rendements moyens des actifs selon les conditions météo.  
5. Tests économétriques et modélisation exploratoire (régression linéaire, Random Forest).

## Évolution du projet

L’analyse a d’abord porté sur la météo à New York afin d’identifier des corrélations statistiques avec plusieurs actifs financiers. Dans un second temps, le projet a évolué vers un focus sur les commodities agricoles, en particulier le maïs, en intégrant des données météorologiques issues des principales zones de production américaines (Corn Belt). L’objectif était d’examiner si les conditions climatiques locales pouvaient être associées aux variations de rendement du future CORN.

## Résultats

Les analyses mettent en évidence certaines corrélations statistiques ponctuelles entre météo et rendements d’actifs, mais celles-ci restent faibles et instables dans le temps. Les modèles prédictifs présentent un pouvoir explicatif limité, en particulier pour la direction journalière des prix.

Concernant le maïs, l’intégration de données météo issues des régions agricoles améliore la cohérence économique de l’approche, mais la météo journalière seule ne permet pas d’expliquer significativement les variations de rendement à court terme.

## Limites

- Corrélation statistique ≠ causalité.  
- Horizon journalier très bruité pour les marchés financiers.  
- Influence d’autres facteurs non intégrés :
  - rapports USDA  
  - exportations  
  - prix de l’énergie  
  - taux de change  
- Utilisation initiale d’une station météo unique.

## Conclusion

Ce projet constitue une exploration quantitative de l’interaction entre variables environnementales et marchés financiers. Il met en évidence la difficulté d’isoler l’impact de la météo sur les rendements journaliers et souligne l’importance d’intégrer des variables économiques, agronomiques et temporelles pour approfondir l’analyse, notamment dans le cas des commodities agricoles comme le maïs.
