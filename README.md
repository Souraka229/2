# DataTour 2026 — Détection fraude Mobile Money

Kit complet pour les étudiants et équipes **DataTour 2026** (Data Afrique Hub).

## Données incluses dans ce dépôt

| Fichier | Description |
|---------|-------------|
| `train.csv` | Entraînement (~1,29 M lignes, cible `fraud_flag`) |
| `test.csv` | Test (~430 k lignes) |
| `sample_submission.csv` | Exemple de format de soumission |
| `datatour_column_descriptions.csv` | Description des colonnes |

## Objectif

Prédire la probabilité de fraude (`target` ∈ [0, 1]) pour chaque transaction.  
Métrique : **Average Precision (PR-AUC)**.

## Installation

```bash
git clone https://github.com/Souraka229/2.git
cd 2
git lfs pull
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

> **Important** : après le clone, exécute `git lfs pull` pour télécharger `train.csv` et `test.csv` (fichiers volumineux via Git LFS).

## Démarrage rapide

**Notebook baseline :**
```bash
jupyter notebook starter_notebook.ipynb
```

**Pipeline avancée :**
```powershell
.\make.ps1 train
```

**Pipeline optimisée v5 :**
```powershell
.\make.ps1 push
```

## Soumission

Fichier `submission.csv` avec colonnes `id,target` — 430 100 lignes, probabilités entre 0 et 1.

## Ressources

- `PLAN_100J.md` — plan d'amélioration
- `starter_notebook.ipynb` — baseline officielle simplifiée
- Plateforme : Data Tour / Data Afrique Hub

Bonne compétition ! #DataTour2026 #CANSD
