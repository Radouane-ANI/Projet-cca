# GEP Mixte : Préconditionnement ILU Incomplet à Précision Mixte Adaptative

[![Matlab](https://img.shields.io/badge/MATLAB-R2021b%2B-blue.svg)](https://www.mathworks.com/products/matlab.html)
[![LaTeX](https://img.shields.io/badge/LaTeX-PDF-green.svg)](https://www.latex-project.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Ce dépôt contient le code source, les scripts d'évaluation et le rapport scientifique complet pour **GEP Mixte**, un solveur linéaire creux et robuste utilisant un préconditionneur ILUT à précision mixte adaptative pour accélérer la convergence de GMRES.

Conçu pour résoudre des systèmes linéaires non symétriques à grande échelle (issus des réseaux électriques, de l'aérospatiale, de la biologie ou de la géophysique), GEP Mixte résout le dilemme entre l'empreinte mémoire et la robustesse du solveur. En allouant dynamiquement l'un des cinq formats de précision (FP16, FP24, FP32, FP48, FP64) à chaque élément individuel du préconditionneur en fonction d'un critère d'erreur locale, GEP Mixte réalise jusqu'à **3.8× de compression mémoire** par rapport à ILUT FP64 traditionnel, sans aucune dégradation de la convergence GMRES.

---

## Structure du Projet

```bash
├── gep_mixte.m                 # Algorithme cœur de GEP Mixte (allocation adaptative)
├── main.m                      # Script de benchmark principal (balayage, Dolan-Moré, tracés)
├── opti_dicho.m                # Algorithme d'optimisation par recherche ternaire (seuil optimal)
├── simuler_precision.m        # Simulateur de précision uniforme (FP64, FP32, FP16, bfloat16 via chop)
├── test_ilu.m                  # Utilitaire d'évaluation GMRES (mesure itérations, résidu, fill-in)
│
├── data/                       # Jeu de données de 6 matrices creuses de la SuiteSparse Collection
│   ├── bcsstk08.mat            # Mécanique des structures (matrice de référence individuelle)
│   ├── 494_bus.mat             # Réseau électrique (Power grid)
│   ├── 1138_bus.mat            # Réseau électrique (Power grid)
│   ├── bp_800.mat              # Simulation de réservoir pétrolier
│   ├── goddardRocketProblem_1.mat # Contrôle optimal (Aérospatiale)
│   └── tumorAntiAngiogenesis_8.mat # Biologie mathématique
│
├── figures_rapport/            # Graphiques générés et inclus dans le rapport LaTeX
├── report.tex                  # Code source LaTeX du rapport académique complet
├── report.pdf                  # Version compilée finale du rapport scientifique (15 pages)
└── README.md                   # Ce fichier de documentation
```

---

## Fondements Mathématiques & Critère Adaptatif

Le critère de GEP Mixte repose sur l'idée que, puisqu'une tolérance de chute $\varepsilon$ est déjà acceptée pour simplifier le préconditionneur (en éliminant les petits éléments), on peut tolérer une erreur d'arrondi de représentation numérique du même ordre de grandeur sur les éléments conservés.

Pour chaque élément non nul du préconditionneur, la précision cible $\tau_{ij}$ est calculée de la manière suivante :
* **Pour les éléments du facteur supérieur $U$ ($u_{ij}$) :**
  $$\tau_{ij} = \frac{\varepsilon \|A_{\cdot,j}\|_2}{|u_{ij}|}$$
* **Pour les éléments du facteur inférieur $L$ ($l_{ij}$) :**
  $$\tau_{ij} = \frac{\varepsilon \|A_{\cdot,j}\|_2}{|l_{ij}| |u_{jj}|}$$

Cette valeur $\tau_{ij}$ est ensuite comparée aux unités d'arrondi des différents formats disponibles ($u_{16}$, $u_{24}$, $u_{32}$, $u_{48}$) afin d'allouer dynamiquement le format de stockage le plus compact possible garantissant la convergence.

---

## Installation & Prérequis

### 1. Environnement MATLAB ou Octave
Le projet est entièrement codé en MATLAB (compatible avec GNU Octave). 

### 2. Bibliothèque de Simulation de Précision `chop`
Pour simuler les formats de précision inférieure (FP16, bfloat16, FP24, FP48), nous utilisons la bibliothèque MATLAB `chop` de Nicholas J. Higham. 
* Si `chop` n'est pas détecté, assurez-vous de l'ajouter dans votre `path` MATLAB ou de l'installer dans un dossier nommé `~/chop` comme configuré au début de `main.m`.

---

## Guide d'Utilisation

### Exécution du Benchmark Complet
Pour lancer la simulation, exécuter l'analyse de sensibilité sur `bcsstk08`, générer le profil de performance multi-matrices de Dolan-Moré et sauvegarder les graphiques :
```matlab
% Dans la console MATLAB :
run main.m
```

### Optimisation par Recherche Ternaire (Dichotomie)
Pour trouver automatiquement la tolérance de chute $\varepsilon$ qui minimise le coût composite (itérations $\times$ fill-in) d'un système linéaire :
```matlab
% Charger votre matrice
load('data/bcsstk08.mat');
M = Problem.A; % ou A selon la matrice

% Lancer l'optimiseur entre epsilon=10^0 et 10^-10 avec une tolérance de 1e-2
[resultats, epsilon_optimal] = opti_dicho(M, 0, 10, 1e-2);
fprintf('Seuil optimal : %e\n', epsilon_optimal);
```

### Compilation du Rapport LaTeX
Pour recompiler le rapport académique sous Linux :
```bash
pdflatex report.tex
pdflatex report.tex  # Une seconde fois pour résoudre les références de figures
```

---

## Principaux Résultats

* **Compression Mémoire Élevée :** GEP Mixte réduit l'empreinte mémoire d'un facteur allant jusqu'à **3.8×** par rapport à une version standard double précision (FP64), tout en conservant une convergence mathématiquement identique.
* **Robustesse Infaillible :** Contrairement à une approche uniforme en basse précision (comme ILUT FP16 qui échoue à converger sur la moitié du jeu de données en raison de l'ill-conditionnement), GEP Mixte atteint **100 % de convergence** sur l'ensemble de la suite SuiteSparse.
* **Complexité Algorithmique Légère :** La sélection dynamique de la précision ne requiert qu'un surcoût négligeable de $O(\text{nnz}(L) + \text{nnz}(U))$, correspondant à seulement **1 multiplication et 1 division** par élément stocké.

---

## Licence

Ce projet est distribué sous licence MIT. Voir le fichier de licence pour plus d'informations.
