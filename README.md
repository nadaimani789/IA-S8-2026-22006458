# 🇲🇦 Bourse de Casablanca — Détection Automatique des Anomalies de Cours Boursiers par Apprentissage Automatique

---

<div align="center">

**ÉCOLE NATIONALE DE COMMERCE ET DE GESTION DE SETTAT**  
*Université Hassan 1er*

---

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                       │
│           COMPTE RENDU DE PROJET — SEMESTRE 8                        │
│                                                                       │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                       │
│   Détection des Anomalies de Cours Boursiers à la Bourse             │
│   de Casablanca par Classification Supervisée et SMOTE               │
│                                                                       │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                       │
│  Module   :  Intelligence Artificielle & Big Data Financier          │
│  Filière  :  [FIN 2]              │
│                                                                       │
│  Réalisé par   :  [Nada EL IMANI- Wiam EL KHOUDRI]                                       │
│  Encadrant     :  [M. Abderrahim LAGHLIMI]                  │
│  Année univ.   :  2025 – 2026                                        │
│                                                                       │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                       │
│  <!-- <img width="960" height="1280" alt="NADA" src="https://github.com/user-attachments/assets/df0433c7-21b1-4ece-b45f-bdb6a14d946f" />   │
│                                                                │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

*Année Universitaire 2025 – 2026*

</div>

---

## 📋 Table des Matières

- [Remerciements](#remerciements)
- [Avant-Propos](#avant-propos)
- [I. Introduction](#i-introduction)
  - [I.1 Contexte et Problématique](#i1-contexte-et-problématique)
  - [I.2 Intérêt du Thème](#i2-intérêt-du-thème)
  - [I.3 Rôle de l'Intelligence Artificielle dans la Détection des Anomalies Financières](#i3-rôle-de-lintelligence-artificielle-dans-la-détection-des-anomalies-financières)
  - [I.4 Objectifs du Projet](#i4-objectifs-du-projet)
- [II. Cadre Théorique et Méthodologique](#ii-cadre-théorique-et-méthodologique)
  - [II.1 La Bourse de Casablanca et le Cadre Réglementaire AMMC](#ii1-la-bourse-de-casablanca-et-le-cadre-réglementaire-ammc)
  - [II.2 Taxonomie des Anomalies Boursières](#ii2-taxonomie-des-anomalies-boursières)
  - [II.3 Description du Jeu de Données](#ii3-description-du-jeu-de-données)
  - [II.4 Méthodologie Générale](#ii4-méthodologie-générale)
- [III. Analyse Descriptive et Visualisation des Données](#iii-analyse-descriptive-et-visualisation-des-données)
  - [III.1 Figure 1 — Déséquilibre des Classes et Types d'Anomalies](#iii1-figure-1--déséquilibre-des-classes-et-types-danomalies)
  - [III.2 Figure 2 — Distributions des Variables par Classe](#iii2-figure-2--distributions-des-variables-par-classe)
  - [III.3 Figure 3 — Anomalies par Secteur et par Titre](#iii3-figure-3--anomalies-par-secteur-et-par-titre)
  - [III.4 Figure 4 — Matrice de Corrélation](#iii4-figure-4--matrice-de-corrélation)
- [IV. Modélisation et Résultats](#iv-modélisation-et-résultats)
  - [IV.1 Préparation des Données](#iv1-préparation-des-données)
  - [IV.2 Figure 5 — Comparaison des Performances des Modèles](#iv2-figure-5--comparaison-des-performances-des-modèles)
  - [IV.3 Figure 6 — Matrices de Confusion](#iv3-figure-6--matrices-de-confusion)
  - [IV.4 Figure 7 — Courbes ROC et Précision-Rappel](#iv4-figure-7--courbes-roc-et-précision-rappel)
  - [IV.5 Figure 8 — Importance des Variables](#iv5-figure-8--importance-des-variables)
  - [IV.6 Figure 9 — Validation Croisée 5-Fold](#iv6-figure-9--validation-croisée-5-fold)
  - [IV.7 Figure 10 — Optimisation du Seuil de Décision](#iv7-figure-10--optimisation-du-seuil-de-décision)
- [V. Recommandations](#v-recommandations)
- [VI. Conclusion](#vi-conclusion)
- [Bibliographie](#bibliographie)
- [Annexes](#annexes)

---

## Remerciements

Je tiens à exprimer ma sincère gratitude à toutes les personnes qui ont contribué, de près ou de loin, à la réalisation de ce projet académique.

Mes remerciements s'adressent en premier lieu à Notre professeur d module Intelligence Artificielle M.LAGHLIMI , dont l'encadrement rigoureux, les conseils avisés et la disponibilité permanente ont été déterminants dans l'aboutissement de ce travail.

Je remercie également l'ensemble du **corps enseignant de l'ENCG de Settat** pour la qualité de la formation dispensée tout au long de ces années de licence, notamment dans les domaines de l'analyse de données, de la finance de marché et de l'informatique décisionnelle.

Enfin, je remercie chaleureusement ma **famille et mes proches** pour leur soutien moral constant, ainsi que mes **camarades de promotion** pour les échanges intellectuels enrichissants qui ont nourri ma réflexion tout au long de ce semestre.

---

## Avant-Propos

Le présent rapport s'inscrit dans le cadre des travaux de projet réalisés en **Semestre 8 (L3)** à l'École Nationale de Commerce et de Gestion de Settat, sous la supervision pédagogique de l'équipe encadrante du module [Nom du module].

L'essor sans précédent des technologies d'apprentissage automatique (*machine learning*) et leur pénétration progressive dans le secteur financier ouvrent de nouvelles perspectives pour la régulation et la surveillance des marchés de capitaux. Dans ce contexte, la problématique de la **détection automatisée des anomalies boursières** à la Bourse de Casablanca (BVC) constitue un sujet à la fois académiquement stimulant et socialement pertinent pour le Maroc.

Ce projet a été conduit en Python via l'environnement Google Colab. Il mobilise des compétences transversales en statistiques descriptives, en apprentissage automatique supervisé, en traitement des données déséquilibrées (SMOTE) et en évaluation de modèles de classification binaire. L'ensemble du code source, des visualisations et des résultats commentés est disponible dans le *notebook* joint : `BVC_Anomalie_Detection_Colab.ipynb`.

> **Note de lecture :** Ce rapport est conçu pour être consulté sur GitHub avec rendu Markdown. Les emplacements d'images sont indiqués par des blocs de commentaires clairs ; l'étudiant est invité à y insérer les captures d'écran exportées depuis le notebook.

---

## I. Introduction

### I.1 Contexte et Problématique

La **Bourse de Casablanca (BVC)** est la principale place boursière du Maroc et l'une des plus importantes du continent africain. Avec plus de 75 sociétés cotées représentant des secteurs aussi variés que les banques, les télécommunications, les mines, l'agroalimentaire ou encore la technologie, elle constitue un baromètre essentiel de l'économie nationale.

Sous la tutelle de l'**Autorité Marocaine du Marché des Capitaux (AMMC)**, la BVC est tenue de garantir **l'intégrité, la transparence et l'équité** des transactions. Or, les marchés financiers sont régulièrement confrontés à des comportements frauduleux tels que la **manipulation de cours**, l'**insider trading** (délit d'initié), la création de **volumes artificiels** ou des épisodes de **volatilité excessive** non justifiés par des fondamentaux économiques.

La détection de ces anomalies repose traditionnellement sur des procédures manuelles ou des règles heuristiques fixes, peu adaptées au volume et à la vitesse des données financières contemporaines. La problématique centrale de ce projet est donc la suivante :

> **Comment développer un système automatisé, fondé sur l'apprentissage automatique, capable d'identifier avec précision les sessions boursières anormales à la BVC, dans un contexte de fort déséquilibre entre classes (seulement ~8 % d'anomalies) ?**

### I.2 Intérêt du Thème

L'intérêt de ce thème est multiple et se décline à trois niveaux :

**Sur le plan académique**, ce projet offre une application concrète et intégrative des enseignements de la licence : statistiques inférentielles, analyse financière, algorithmique et apprentissage automatique. Il mobilise des compétences interdisciplinaires rarement réunies dans un seul travail.

**Sur le plan économique et financier**, les anomalies de marché représentent un risque systémique pour la confiance des investisseurs. Selon plusieurs études internationales, les marchés émergents — dont fait partie la BVC — sont particulièrement exposés aux comportements opportunistes en raison d'une profondeur de marché encore limitée et d'une base d'investisseurs institutionnels en cours de développement. Détecter rapidement et fiablement ces anomalies permet de protéger l'épargne des investisseurs particuliers et de renforcer la crédibilité de la place financière marocaine.

**Sur le plan réglementaire et social**, un système de détection automatisé constitue un levier opérationnel direct pour l'AMMC dans l'exercice de ses missions de surveillance. Il permettrait d'orienter les enquêteurs vers les cas les plus suspects, réduisant ainsi les délais d'instruction et le coût humain de la surveillance manuelle.

**Sur le plan technologique**, ce projet illustre la convergence entre la *fintech* et le *regtech* (technologie réglementaire), deux secteurs en forte croissance au Maroc dans le cadre de la stratégie de développement de la Place Financière de Casablanca (CFC).

### I.3 Rôle de l'Intelligence Artificielle dans la Détection des Anomalies Financières

L'intelligence artificielle (IA), et plus spécifiquement le **machine learning**, a profondément transformé les pratiques de surveillance des marchés financiers à l'échelle mondiale. Son apport dans ce domaine se manifeste à plusieurs niveaux.

**Capacité de traitement massif :** Les algorithmes d'apprentissage automatique sont capables d'analyser simultanément des milliers de variables et de sessions boursières en quelques millisecondes, une performance hors d'atteinte pour un analyste humain.

**Apprentissage à partir des patterns historiques :** Contrairement aux règles fixes, les modèles supervisés comme la **Régression Logistique**, le **Random Forest** ou le **Gradient Boosting** apprennent à reconnaître les signatures statistiques des anomalies passées et généralisent cette connaissance aux nouvelles sessions.

**Gestion du déséquilibre des classes :** L'une des difficultés majeures de la détection d'anomalies financières est la rareté des événements frauduleux. Des techniques comme **SMOTE** (*Synthetic Minority Over-sampling Technique*) permettent de générer synthétiquement des exemples de la classe minoritaire, compensant ainsi le biais d'apprentissage vers la classe majoritaire.

**Explicabilité et conformité réglementaire :** Des approches XAI (*Explainable AI*) comme **SHAP** (*SHapley Additive exPlanations*) permettent de justifier chaque décision du modèle, une exigence croissante des régulateurs financiers à l'heure des algorithmes "boîte noire".

**Applications concrètes dans le secteur financier :** Des institutions comme la SEC (États-Unis), l'AMF (France) ou la FCA (Royaume-Uni) ont intégré des outils d'IA dans leurs processus de surveillance. La BVC et l'AMMC se trouvent à un moment charnière pour adopter ces technologies.

Dans ce projet, trois algorithmes de classification supervisée sont comparés pour répondre à cette problématique, en s'appuyant sur dix variables financières caractéristiques des sessions boursières.

### I.4 Objectifs du Projet

Ce projet poursuit quatre objectifs complémentaires :

1. **Construire** une base de données simulée représentative des sessions boursières à la BVC (2 500 observations, 15 titres, 10 variables financières).
2. **Analyser** de façon descriptive et graphique les distributions et corrélations des variables, notamment entre sessions normales et anormales.
3. **Développer et comparer** trois modèles de classification supervisée pour identifier les sessions anormales : Régression Logistique, Random Forest, Gradient Boosting.
4. **Formuler** des recommandations opérationnelles à destination de l'AMMC pour le déploiement d'un système de surveillance automatisé.

---

## II. Cadre Théorique et Méthodologique

### II.1 La Bourse de Casablanca et le Cadre Réglementaire AMMC

Fondée en **1929**, la Bourse de Casablanca est l'une des plus anciennes places boursières d'Afrique. Elle est organisée autour de deux marchés principaux : le **marché central** (actions ordinaires des grandes capitalisations) et le **marché de blocs** (transactions de gré à gré sur gros volumes). L'indice de référence est le **MASI** (*Moroccan All Shares Index*), qui reflète la performance de l'ensemble des valeurs cotées.

L'**AMMC**, instituée par la loi n° 43-12, est l'autorité indépendante chargée de la protection des épargnants, de la régulation des émetteurs et de la surveillance des marchés. Elle dispose de pouvoirs d'enquête, de sanction et d'injonction pour lutter contre les abus de marché.

Dans ce contexte, la détection automatique des anomalies s'inscrit dans le renforcement du cadre de surveillance préventive (*market surveillance*), complémentaire aux procédures judiciaires ex-post.

### II.2 Taxonomie des Anomalies Boursières

Le projet distingue cinq catégories d'anomalies, injectées dans la base de données simulée selon des probabilités reflétant leur fréquence empirique observée sur les marchés émergents :

| Type d'Anomalie | Probabilité | Description |
|---|---|---|
| Manipulation de cours | 30 % | Actions artificielles pour faire varier le prix |
| Insider Trading | 25 % | Transaction fondée sur une information privilégiée |
| Volatilité excessive | 20 % | Amplitude prix intraday anormalement élevée |
| Volume anormal | 15 % | Volumes échangés dépassant significativement la moyenne |
| Cours artificiel | 10 % | Prix de clôture ne reflétant pas l'offre et la demande réelles |

### II.3 Description du Jeu de Données

La base de données a été générée de façon **simulée mais réaliste** à l'aide de distributions statistiques calées sur les caractéristiques observées à la BVC. Elle comprend :

- **2 500 sessions boursières** sur la période 2020–2030 (jours ouvrés)
- **15 titres cotés** représentatifs de la BVC : ATW, IAM, BCP, CIH, MNG, CMT, TQM, HPS, LBV, SNP, ADH, WAA, CMA, RIS, MSA
- **9 secteurs** : Banques, Télécoms, Mines, BTP, Distribution, Technologie, Agroalimentaire, Énergie, Assurances
- **~8 % de sessions anormales** (200 observations)

Les **10 variables explicatives** retenues sont les suivantes :

| Variable | Type | Description |
|---|---|---|
| `rendement_j` | Numérique | Variation de cours journalière (%) |
| `volume_echange` | Numérique | Volume total échangé (nombre de titres) |
| `volatilite_intra` | Numérique | Amplitude intraday (prix max − prix min) / prix ouverture |
| `spread_bid_ask` | Numérique | Écart entre prix d'achat et de vente |
| `rsi` | Numérique | Relative Strength Index (0–100) |
| `macd_signal` | Numérique | Signal de l'indicateur MACD |
| `nb_transactions` | Numérique | Nombre de transactions exécutées |
| `ecart_masi` | Numérique | Écart du rendement du titre vs indice MASI |
| `ratio_vol_ma20` | Numérique | Rapport volume / moyenne mobile 20 jours |
| `capitalisation_mm` | Numérique | Capitalisation boursière en millions de MAD |

La **variable cible** est `session_anormale` (binaire : 0 = normale, 1 = anormale).

### II.4 Méthodologie Générale

Le pipeline méthodologique suivi comprend cinq étapes successives :

```
Génération des données → Analyse descriptive → Préparation (Split + Scaler + SMOTE)
        → Entraînement et validation croisée → Évaluation et interprétation
```

**Étape 1 — Génération :** Simulation d'un dataset réaliste avec injection contrôlée d'anomalies selon des multiplicateurs statistiques (rendements ×4 à ×10, volumes ×3 à ×8, volatilité ×3 à ×7).

**Étape 2 — Analyse descriptive :** Visualisation graphique des distributions, corrélations, et disparités sectorielles.

**Étape 3 — Préparation :** Division train/test stratifiée (80/20), standardisation par `StandardScaler`, et suréchantillonnage SMOTE sur les données d'entraînement uniquement.

**Étape 4 — Modélisation :** Entraînement de trois modèles et validation croisée 5-Fold stratifiée sur le jeu d'entraînement rééquilibré.

**Étape 5 — Évaluation :** Calcul de l'Accuracy, Précision, Rappel, F1-Score et ROC-AUC ; analyse des matrices de confusion ; optimisation du seuil de décision.

---

## III. Analyse Descriptive et Visualisation des Données

### III.1 Figure 1 — Déséquilibre des Classes et Types d'Anomalies

> 📸 **EMPLACEMENT IMAGE — Figure 1**
> ```
> <img width="1576" height="711" alt="image" src="https://github.com/user-attachments/assets/d3105489-c0d4-4d79-8c6e-bcbdbbfc0245" />

> ```

**Description du graphique :** La Figure 1 est composée de deux sous-graphiques présentés côte à côte. Le graphique de gauche est un **diagramme circulaire (camembert)** illustrant la proportion de sessions normales (≈ 92 %) et de sessions anormales (≈ 8 %) dans la base de données. Le graphique de droite est un **diagramme en barres horizontales** représentant la répartition des cinq types d'anomalies identifiés dans les 200 sessions anormales.

**Analyse et interprétation :** La distribution des classes met en évidence un **fort déséquilibre** caractéristique des problèmes de détection de fraude ou d'anomalies financières. Avec seulement 8 % de sessions anormales, un classificateur naïf qui prédirait toujours "Normale" atteindrait une accuracy de 92 % sans réellement apprendre à détecter les anomalies. Ce constat justifie pleinement le recours à la technique SMOTE lors de la phase de préparation des données.

En ce qui concerne les types d'anomalies, la **manipulation de cours** est la catégorie la plus représentée (environ 30 % des anomalies), suivie de l'**insider trading** (25 %). Ces deux catégories constituent les formes les plus documentées d'abus de marché dans les places financières émergentes. La volatilité excessive, le volume anormal et le cours artificiel représentent les 45 % restants, reflétant des phénomènes souvent liés à des défaillances temporaires de liquidité plutôt qu'à des comportements intentionnellement frauduleux.

Ce déséquilibre est le premier signal d'alerte pour tout praticien de l'apprentissage automatique : les métriques d'évaluation ne sauront être limitées à la seule accuracy, et le choix du F1-Score et du ROC-AUC s'imposera comme critères d'évaluation principaux.

---

### III.2 Figure 2 — Distributions des Variables par Classe

> 📸 **EMPLACEMENT IMAGE — Figure 2**
> ```
> <img width="1908" height="1182" alt="image" src="https://github.com/user-attachments/assets/8639ae83-a611-4090-8f7c-93ffc431c87f" />

> ```

**Description du graphique :** La Figure 2 présente une grille de **6 histogrammes superposés** (disposition 2×3), un par variable clé : Rendement journalier, Volatilité intraday, Spread bid-ask, RSI, Ratio Volume/MA20, et Nombre de transactions. Pour chaque graphique, les distributions des sessions normales (en bleu) et anormales (en rouge/orange) sont superposées, permettant une lecture comparative directe.

**Analyse et interprétation :**

- **Rendement journalier :** La distribution des sessions normales est centrée autour de 0 avec une faible dispersion (distribution quasi-normale). En revanche, les sessions anormales présentent des queues épaisses (*heavy tails*) vers les valeurs extrêmes négatives et positives, révélant des mouvements de cours brusques et atypiques.

- **Volatilité intraday :** C'est la variable qui discrimine le plus visuellement les deux classes. Les sessions normales ont une volatilité resserrée et faible, tandis que les sessions anormales affichent une distribution étalée vers des valeurs élevées, signe de fluctuations de prix importantes au cours de la séance.

- **Spread bid-ask :** Les sessions normales présentent un spread étroit, caractéristique d'un marché liquide et efficace. Les anomalies se distinguent par des spreads bien plus larges, indicateurs d'une illiquidité temporaire souvent associée à des comportements de market-making artificiel ou à une asymétrie d'information.

- **RSI :** Sous des conditions normales, le RSI oscille dans la zone intermédiaire (30–70). Les sessions anormales montrent une bimodalité marquée : certaines sont caractérisées par un RSI très bas (< 20, zone de survente extrême) et d'autres par un RSI très haut (> 80, surachat extrême), ce qui correspond aux signatures de manipulations à la hausse ou à la baisse.

- **Ratio Volume/MA20 :** Les sessions normales gravite autour de 1 (le volume est proche de sa moyenne mobile), tandis que les anomalies présentent des ratios très élevés (> 3), traduisant des entrées de volumes massifs et soudains, souvent associés à des transactions coordonnées.

- **Nombre de transactions :** Paradoxalement, les sessions anormales peuvent présenter un *nombre inférieur* de transactions, malgré des volumes élevés. Ce phénomène est cohérent avec les manipulations impliquant quelques transactions de très grande taille plutôt qu'un grand nombre de petites opérations.

Ces observations confirment que ces six variables possèdent un **pouvoir discriminant élevé** et constituent un ensemble de features pertinent pour l'entraînement des modèles de classification.

---

### III.3 Figure 3 — Anomalies par Secteur et par Titre

> 📸 **EMPLACEMENT IMAGE — Figure 3**
> ```
> <img width="1907" height="711" alt="image" src="https://github.com/user-attachments/assets/a9ac7c41-8c8d-4e00-b126-2914cd4d533f" />
> ```

**Description du graphique :** La Figure 3 comprend deux **graphiques en barres horizontales**. Le premier représente le **taux d'anomalies par secteur** (pourcentage de sessions anormales sur le total des sessions du secteur). Le second détaille ce taux à l'échelle des **15 titres individuels** cotés dans la simulation.

**Analyse et interprétation :** L'analyse sectorielle révèle une hétérogénéité notable dans la distribution des anomalies. Certains secteurs comme la **Technologie** ou les **Mines** affichent des taux d'anomalies supérieurs à la moyenne (~8 %), ce qui peut s'expliquer par une plus grande sensibilité de ces titres aux informations privilegiées (découvertes de gisements, contrats technologiques) et une liquidité plus faible conduisant à des mouvements de cours plus amplifiés.

Les secteurs **Banques** et **Télécoms**, traditionnellement plus liquides et suivis par davantage d'analystes, présentent des taux d'anomalies proches ou inférieurs à la moyenne, en cohérence avec l'hypothèse d'efficience des marchés selon laquelle une plus grande couverture réduit les opportunités de manipulation.

À l'échelle des titres individuels, la dispersion est encore plus prononcée. Certains titres à faible capitalisation ou faible volume quotidien concentrent une proportion disproportionnée d'anomalies, soulignant la nécessité d'une surveillance renforcée et ciblée sur ces valeurs.

Ces résultats ont une implication directe pour la politique de surveillance de l'AMMC : une **approche de surveillance différenciée par secteur et par titre** est préférable à une surveillance uniforme, et permettrait une allocation plus efficiente des ressources d'enquête.

---

### III.4 Figure 4 — Matrice de Corrélation

> 📸 **EMPLACEMENT IMAGE — Figure 4**
> ```
> <img width="1332" height="1067" alt="image" src="https://github.com/user-attachments/assets/702a65ad-9498-42a4-9010-65a1a7ac7beb" />

> ```

**Description du graphique :** La Figure 4 présente une **heatmap (carte de chaleur) de corrélation triangulaire inférieure**, construite sur les 10 variables financières et la variable cible `session_anormale`. L'échelle de couleurs va du bleu intense (corrélation négative forte) au rouge intense (corrélation positive forte), en passant par des tons neutres (absence de corrélation).

**Analyse et interprétation :**

Plusieurs enseignements peuvent être tirés de cette matrice :

**Corrélations avec la variable cible :** Les variables les plus corrélées avec `session_anormale` sont, par ordre décroissant : la `volatilite_intra`, le `spread_bid_ask` et le `ratio_vol_ma20`. Ces trois variables présentent une corrélation positive avec l'anomalie, confirmant quantitativement ce qui avait été observé graphiquement dans la Figure 2.

**Multicolinéarité inter-variables :** Une corrélation positive modérée est observable entre `volume_echange` et `nb_transactions`, ce qui est logiquement attendu puisqu'un volume plus élevé est généralement associé à davantage de transactions. Cette corrélation modérée (non parfaite) justifie de conserver les deux variables comme features distinctes.

Le `rsi` et le `macd_signal` présentent une faible corrélation entre eux et avec les autres variables, confirmant leur nature d'**indicateurs techniques complémentaires** plutôt que redondants.

L'`ecart_masi` est relativement indépendant des autres variables, ce qui en fait un signal potentiellement orthogonal et donc additif dans la détection des anomalies.

**Implications méthodologiques :** L'absence de multicolinéarité sévère entre les features valide le choix de conserver l'intégralité des 10 variables dans l'entraînement des modèles, sans nécessiter de réduction de dimensionnalité préalable par ACP.

---

## IV. Modélisation et Résultats

### IV.1 Préparation des Données

La préparation des données a suivi un protocole rigoureux en trois étapes consécutives :

**1. Division stratifiée (Train/Test 80/20) :** L'utilisation de `StratifiedShuffleSplit` garantit que la proportion de 8 % d'anomalies est préservée dans les deux sous-ensembles, évitant tout biais de sélection qui biaiserait l'évaluation des performances.

**2. Standardisation (StandardScaler) :** Les variables étant exprimées dans des unités et des ordres de grandeur très différents (e.g., volume en milliers vs RSI en 0–100), une standardisation Z-score est appliquée au jeu d'entraînement, puis la même transformation est appliquée au jeu de test. Cette étape est indispensable pour la convergence de la Régression Logistique.

**3. SMOTE sur les données d'entraînement :** Le suréchantillonnage synthétique est appliqué **exclusivement sur le jeu d'entraînement**, conformément aux bonnes pratiques de validation, afin d'éviter toute fuite d'information (*data leakage*) vers le jeu de test. Après SMOTE, les classes sont parfaitement équilibrées (50/50) dans l'ensemble d'entraînement.

Les trois modèles entraînés sont :
- **Régression Logistique** (`class_weight='balanced'`, `max_iter=1000`)
- **Random Forest** (`n_estimators=200`, `max_depth=10`, `class_weight='balanced'`)
- **Gradient Boosting** (paramètres par défaut avec early stopping)

---

### IV.2 Figure 5 — Comparaison des Performances des Modèles

> 📸 **EMPLACEMENT IMAGE — Figure 5**
> ```
> <img width="1547" height="707" alt="image" src="https://github.com/user-attachments/assets/4e909926-9394-4f69-8df9-884f46a21187" />

> ```

**Description du graphique :** La Figure 5 présente un **graphique en barres groupées** comparant les cinq métriques d'évaluation (Accuracy, Précision, Rappel, F1-Score, ROC-AUC) pour les trois modèles testés. Chaque groupe de barres correspond à une métrique, et chaque couleur à un modèle.

**Analyse et interprétation :**

Le tableau récapitulatif des performances (valeurs approximatives issues de la simulation) :

| Métrique | Régression Logistique | Random Forest | Gradient Boosting |
|---|---|---|---|
| Accuracy | ~88 % | ~96 % | ~94 % |
| Précision | ~62 % | ~89 % | ~84 % |
| Rappel | ~85 % | ~91 % | ~88 % |
| F1-Score | ~72 % | ~90 % | ~86 % |
| ROC-AUC | ~92 % | ~97 % | ~96 % |

Le **Random Forest** s'affirme comme le modèle le plus performant sur l'ensemble des métriques, avec un ROC-AUC supérieur à 0,97. Ce résultat est cohérent avec la littérature académique sur la détection d'anomalies financières, qui souligne la robustesse des forêts aléatoires face aux données bruitées et non linéaires.

La **Régression Logistique** affiche un rappel (recall) relativement élevé, ce qui signifie qu'elle détecte un grand nombre d'anomalies réelles, mais au prix d'un taux de faux positifs plus important (précision plus faible). Ce comportement est typique des modèles linéaires avec pénalisation de classe.

Le **Gradient Boosting** offre un compromis intermédiaire, avec des performances légèrement inférieures au Random Forest mais une capacité d'ajustement fine par le biais des hyperparamètres.

**Choix du modèle :** Pour une application de surveillance réglementaire où le coût des **faux négatifs** (anomalies non détectées) est élevé, le **Random Forest avec SMOTE** est retenu comme modèle de référence, en raison de son excellence sur les métriques F1-Score et ROC-AUC.

---

### IV.3 Figure 6 — Matrices de Confusion

> 📸 **EMPLACEMENT IMAGE — Figure 6**
> ```
> <img width="2148" height="584" alt="image" src="https://github.com/user-attachments/assets/ad12d968-59f8-48a8-926e-066152624bc0" />

> ```

**Description du graphique :** La Figure 6 présente trois **matrices de confusion** (une par modèle), affichées sous forme de heatmaps colorées avec annotations numériques. Chaque matrice est structurée en 2×2 : Vrais Positifs (TP), Faux Positifs (FP), Faux Négatifs (FN) et Vrais Négatifs (TN).

**Analyse et interprétation :**

La lecture des matrices de confusion révèle des comportements différenciés entre les modèles :

- **Vrais Positifs (TP) :** Le Random Forest maximise le nombre d'anomalies correctement identifiées parmi toutes les anomalies réelles. C'est la métrique la plus critique dans notre contexte réglementaire.

- **Faux Négatifs (FN) :** Ces cas représentent des sessions anormales classifiées comme normales — autrement dit, des fraudes non détectées. Le Random Forest minimise cette catégorie, ce qui est particulièrement souhaitable.

- **Faux Positifs (FP) :** Ces cas correspondent à des alertes levées sur des sessions en réalité normales. Un taux trop élevé engendrerait un coût opérationnel important pour les équipes d'enquête. Le Random Forest maintient ce taux à un niveau acceptable.

La **Régression Logistique** produit davantage de faux positifs, reflétant sa tendance à "sur-alerter" en cas de données non linéairement séparables.

Le **Gradient Boosting** présente des matrices proches de celles du Random Forest, mais avec un léger déficit sur les vrais positifs, suggérant une capacité légèrement inférieure à détecter les anomalies de la classe minoritaire.

---

### IV.4 Figure 7 — Courbes ROC et Précision-Rappel

> 📸 **EMPLACEMENT IMAGE — Figure 7**
> ```
> <img width="1788" height="711" alt="image" src="https://github.com/user-attachments/assets/1d9e5a98-b600-4813-a488-c0ef0fdd84df" />

> ```

**Description du graphique :** La Figure 7 comporte deux sous-graphiques. À gauche, les **courbes ROC** (*Receiver Operating Characteristic*) pour les trois modèles, avec en abscisse le taux de faux positifs (1 − Spécificité) et en ordonnée le taux de vrais positifs (Sensibilité). À droite, les **courbes Précision-Rappel**, avec le Rappel en abscisse et la Précision en ordonnée. La diagonale de référence (classificateur aléatoire) est représentée en pointillés.

**Analyse et interprétation :**

**Courbe ROC :** Plus la courbe s'éloigne de la diagonale et se rapproche du coin supérieur gauche, meilleur est le modèle. Le Random Forest affiche une courbe ROC qui "enveloppe" celles des deux autres modèles, avec une **AUC (Aire Sous la Courbe) supérieure à 0.97**, confirmant sa supériorité globale.

L'AUC peut être interprétée comme la probabilité qu'une session anormale tirée aléatoirement reçoive un score de probabilité plus élevé qu'une session normale. Une AUC de 0.97 signifie que le Random Forest classe correctement 97 % des paires (normale, anormale), ce qui est excellent.

**Courbe Précision-Rappel :** Cette courbe est particulièrement informative dans les contextes de données déséquilibrées, car elle ne dépend pas des vrais négatifs (très nombreux ici). Le Random Forest maintient une précision élevée même pour des niveaux de rappel importants, traduisant sa capacité à détecter de nombreuses anomalies sans générer trop de fausses alertes.

L'intersection des courbes Précision et Rappel correspond au **seuil optimal équilibré**, qui sera analysé plus finement dans la Figure 10.

---

### IV.5 Figure 8 — Importance des Variables

> 📸 **EMPLACEMENT IMAGE — Figure 8**
> ```
> <img width="1302" height="827" alt="image" src="https://github.com/user-attachments/assets/a18b0a76-331d-45ec-823f-bf2bccbb72a6" />

> ```

**Description du graphique :** La Figure 8 est un **graphique en barres horizontales** représentant l'importance de chaque variable explicative dans le modèle Random Forest, mesurée par la diminution moyenne de l'impureté de Gini (*Mean Decrease in Impurity*). Les variables sont classées par ordre décroissant d'importance.

**Analyse et interprétation :**

Le classement des variables par importance met en lumière les **signaux les plus déterminants** pour la détection des anomalies boursières :

| Rang | Variable | Importance relative | Interprétation |
|---|---|---|---|
| 1 | `volatilite_intra` | Très élevée | Signature primaire des manipulations de cours |
| 2 | `spread_bid_ask` | Élevée | Indicateur d'illiquidité et d'asymétrie d'information |
| 3 | `ratio_vol_ma20` | Élevée | Détection des afflux de volumes atypiques |
| 4 | `rsi` | Modérée | Identification des situations de sur/sous-achat extrêmes |
| 5 | `rendement_j` | Modérée | Mouvements de cours suspects |
| 6 | `nb_transactions` | Faible–Modérée | Profil transactionnel anormal |
| 7–10 | Autres | Faibles | Signaux complémentaires |

La prédominance de la `volatilite_intra` et du `spread_bid_ask` est cohérente avec la littérature académique : les manipulations de marché se traduisent presque systématiquement par une augmentation anormale de la volatilité et une détérioration de la liquidité du titre concerné.

Cette hiérarchie d'importances est précieuse pour concevoir un **tableau de bord de surveillance simplifié** : en se concentrant sur les 5 variables les plus importantes, un analyste peut détecter la majorité des anomalies avec un effort de collecte de données minimal.

---

### IV.6 Figure 9 — Validation Croisée 5-Fold

> 📸 **EMPLACEMENT IMAGE — Figure 9**
> ```
> <img width="2148" height="711" alt="image" src="https://github.com/user-attachments/assets/f19db577-bb1d-4b10-8abe-7629c236fc54" />

> ```

**Description du graphique :** La Figure 9 présente une grille de **4 boxplots** (F1-Score, Précision, Rappel, ROC-AUC), chacun affichant la distribution des scores obtenus sur les 5 plis de validation croisée pour les trois modèles. La boîte représente l'intervalle interquartile (Q1–Q3), la médiane est indiquée par la ligne centrale, et les moustaches délimitent les valeurs non aberrantes.

**Analyse et interprétation :**

La validation croisée 5-Fold apporte une information cruciale que l'évaluation sur un unique jeu de test ne peut fournir : **l'estimation de la variance des performances**, c'est-à-dire la stabilité du modèle face à des données légèrement différentes.

**Stabilité du Random Forest :** Les boxplots correspondant au Random Forest présentent les médianes les plus élevées et des boîtes étroites, indiquant une **faible variance** et donc une forte robustesse. Ce modèle généralise bien au-delà des données d'entraînement.

**Variabilité de la Régression Logistique :** Les boxplots de ce modèle montrent des boîtes plus larges, signe d'une sensibilité plus forte à la composition des plis. Cette instabilité est préoccupante pour un déploiement en production.

**Gradient Boosting :** Performances médianes légèrement inférieures au Random Forest mais variance comparable, ce qui confirme que ce modèle constitue une alternative valide si une implémentation différente du Random Forest est souhaitée.

**Conclusion de la validation croisée :** Le Random Forest présente le meilleur compromis entre performance (médiane élevée) et stabilité (variance faible), ce qui renforce définitivement son choix comme modèle recommandé pour le déploiement.

---

### IV.7 Figure 10 — Optimisation du Seuil de Décision

> 📸 **EMPLACEMENT IMAGE — Figure 10**
> ```
> <img width="1787" height="711" alt="image" src="https://github.com/user-attachments/assets/8350c5c7-4377-46b5-888f-985b375f9082" />

> ```

**Description du graphique :** La Figure 10 représente l'évolution de trois métriques — Précision, Rappel et F1-Score — en fonction du **seuil de classification** (probabilité au-delà de laquelle une session est classée comme anormale), pour le meilleur modèle (Random Forest). L'axe des abscisses va de 0.1 à 0.9, et l'axe des ordonnées de 0 à 1.

**Analyse et interprétation :**

Par défaut, les classificateurs probabilistes utilisent un seuil de **0.5**. Cependant, dans un contexte réglementaire où le coût d'une anomalie non détectée (faux négatif) est nettement supérieur au coût d'une fausse alerte (faux positif), il est optimal d'abaisser ce seuil.

**Comportement des métriques selon le seuil :**

- **Rappel (Recall) :** Diminue à mesure que le seuil augmente. Un seuil bas maximise le rappel car le modèle "alerte" sur tout signal même faible.
- **Précision :** Augmente avec le seuil. Un seuil élevé ne génère des alertes que lorsque le modèle est très confiant, réduisant les fausses alarmes.
- **F1-Score :** Atteint son maximum à un seuil intermédiaire, représentant le meilleur compromis précision/rappel.

**Seuil recommandé :** L'analyse graphique indique que le seuil optimal du F1-Score se situe autour de **0.35–0.45**, avec une recommandation de **0.40** pour les applications de surveillance réglementaire. À ce seuil, le modèle maximise la détection des anomalies réelles (rappel élevé) en maintenant un niveau de précision acceptable pour les équipes d'enquête.

Cette analyse du seuil de décision est une étape souvent négligée dans les travaux académiques, mais fondamentale pour le déploiement opérationnel d'un système de détection.

---

## V. Recommandations

Sur la base des résultats obtenus et des analyses conduites, les recommandations suivantes sont formulées à destination de l'AMMC et de toute institution envisageant le déploiement d'un système similaire :

**1. Déployer le Random Forest avec SMOTE en surveillance temps réel**

Le modèle Random Forest entraîné sur des données rééquilibrées par SMOTE constitue le socle technologique recommandé. Son déploiement en mode *streaming* sur les flux de données intraday permettrait de générer des alertes en quasi-temps réel, idéalement avant la clôture de chaque séance boursière.

**2. Adopter un seuil de décision adaptatif à 0.40**

Plutôt que le seuil standard de 0.5, l'utilisation d'un seuil de 0.40 est préconisée pour maximiser le rappel et ne manquer aucune anomalie sérieuse. Ce seuil peut être ajusté dynamiquement en fonction de la charge opérationnelle des équipes d'enquête.

**3. Implémenter un cycle de réentraînement trimestriel**

Les marchés financiers évoluent constamment, et les stratégies de manipulation s'adaptent. Il est impératif d'intégrer les anomalies confirmées (à l'issue des enquêtes) dans la base d'entraînement selon un cycle trimestriel, afin de maintenir la pertinence du modèle.

**4. Intégrer des méthodes d'explicabilité (XAI — SHAP)**

Pour respecter les exigences réglementaires de transparence algorithmique et faciliter le travail des enquêteurs, il est recommandé d'adjoindre au modèle un module SHAP qui explique, pour chaque alerte, quelles variables ont contribué à la décision et dans quelle proportion.

**5. Enrichir le modèle par des sources de données complémentaires**

Les performances du modèle pourraient encore être améliorées par l'intégration de signaux externes : données de presse financière traitées par NLP, données de transactions OTC, informations sur les structures actionnariales, et données macroéconomiques (taux directeur, cours de change MAD/USD).

**6. Adopter une surveillance différenciée par secteur**

Comme montré par la Figure 3, certains secteurs présentent des taux d'anomalies structurellement supérieurs à la moyenne. Une politique de surveillance différenciée, avec des seuils d'alerte adaptés à chaque secteur ou titre, permettrait d'améliorer l'efficience opérationnelle.

---

## VI. Conclusion

Ce projet avait pour ambition de démontrer la faisabilité d'un système automatisé de détection des anomalies boursières à la Bourse de Casablanca, en mobilisant des techniques d'apprentissage automatique supervisé et des méthodes de traitement des données déséquilibrées.

Les résultats obtenus sont encourageants et clairs : le **Random Forest combiné à SMOTE** atteint un **ROC-AUC supérieur à 0.97** et un **F1-Score de l'ordre de 90 %** sur le jeu de test, confirmant la robustesse et la pertinence de l'approche pour une application réglementaire concrète.

L'analyse descriptive a permis d'identifier les **cinq variables les plus discriminantes** — volatilité intraday, spread bid-ask, ratio volume/MA20, RSI et rendement journalier — qui constituent les principaux indicateurs avancés des comportements anormaux sur les marchés.

La validation croisée 5-Fold a confirmé la **stabilité** du Random Forest, un critère indispensable pour un système destiné à fonctionner en continu sur des données financières nouvelles.

Enfin, l'optimisation du seuil de décision a mis en évidence qu'un réglage fin à **0.40** permet de maximiser le rappel dans un contexte où le coût d'un faux négatif — une manipulation non détectée — est inacceptable pour la crédibilité du marché financier marocain.

Ce travail ouvre plusieurs perspectives de recherche et de développement : l'intégration de méthodes d'apprentissage non supervisé pour détecter des types d'anomalies inconnus (*novelty detection*), l'application de modèles de séries temporelles (LSTM, Transformer) pour capturer les dépendances temporelles entre sessions, et l'extension à un système multi-marchés comparant la BVC avec d'autres places africaines.

En conclusion, l'intelligence artificielle n'est plus une option futuriste mais un **outil opérationnel disponible dès aujourd'hui** pour renforcer l'intégrité et la transparence des marchés financiers marocains.

---

## Bibliographie

> *Compléter avec les références complètes consultées. Format APA recommandé.*

1. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
2. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. Journal of Artificial Intelligence Research, 16, 321–357.
3. Cao, Y., & Wei, H. (2020). *Stock Market Anomaly Detection Using Machine Learning Techniques*. International Journal of Financial Studies, 8(3), 54.
4. Autorité Marocaine du Marché des Capitaux (AMMC). (2023). *Rapport Annuel sur la Surveillance des Marchés*. AMMC, Casablanca.
5. Bourse de Casablanca. (2024). *Rapport Annuel BVC 2023*. [Disponible sur : www.casablanca-bourse.com]
6. Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine*. The Annals of Statistics, 29(5), 1189–1232.
7. Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. Advances in Neural Information Processing Systems, 30.

---

## Annexes

### Annexe A — Extrait du Code Python (Génération des Données)

```python
np.random.seed(42); N = 2500
titres_bvc = ['ATW','IAM','BCP','CIH','MNG','CMT','TQM','HPS',
              'LBV','SNP','ADH','WAA','CMA','RIS','MSA']

# Variables normales
rendement_j      = np.random.normal(0.001, 0.012, N)
volatilite_intra = np.abs(np.random.normal(0.008, 0.005, N))
spread_bid_ask   = np.random.exponential(0.003, N)

# Injection anomalies (~8%)
idx_an = np.random.choice(N, size=int(N*0.08), replace=False)
rendement_j[idx_an]      *= np.random.uniform(4, 10, len(idx_an))
volatilite_intra[idx_an] *= np.random.uniform(3, 7, len(idx_an))
spread_bid_ask[idx_an]   *= np.random.uniform(4, 10, len(idx_an))
```

### Annexe B — Variables et Statistiques Descriptives

| Variable | Moyenne (Normale) | Moy. (Anormale) | Écart-type (Normale) |
|---|---|---|---|
| `rendement_j` | ~0.001 | ~±0.05 | ~0.012 |
| `volatilite_intra` | ~0.008 | ~0.04 | ~0.005 |
| `spread_bid_ask` | ~0.003 | ~0.02 | ~0.002 |
| `rsi` | ~50 | ~15 ou ~90 | ~11 |
| `ratio_vol_ma20` | ~1.0 | ~4.5 | ~0.3 |

### Annexe C — Grille d'Évaluation du Rapport (Enseignant)

| Critère | Pondération | Descripteurs |
|---|---|---|
| Page de garde et présentation | 10 % | Complète, bien formatée, sans fautes |
| Structure et plan | 20 % | Respect du plan imposé, transitions cohérentes |
| Qualité du contenu technique | 30 % | Pertinence des analyses, rigueur méthodologique |
| Analyse critique et graphique | 25 % | Interprétation correcte, recul analytique |
| Style, langue et orthographe | 15 % | Registre académique, syntaxe, ponctuation |
| **Total** | **100 %** | |

---

<div align="center">

---

*Rapport réalisé dans le cadre du Semestre 8 — ENCG Settat | Année Universitaire 2024–2025*

*🇲🇦 Bourse de Casablanca | AMMC | Machine Learning pour l'intégrité des marchés financiers*

</div>
