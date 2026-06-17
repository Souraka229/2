# Plan 100 jours — DataTour 2026 Mobile Money Fraud

**Point de départ** : AP=0.360 (temporel) / 0.400 (group-aware), ensemble LGBM+CatBoost+XGBoost.
**Objectif** : AP ≥ 0.55, top mondial, **zéro fuite de cible**, robustesse prouvée par CV.
**Règle stricte anti-triche** : aucun feature qui touche `fraud_flag` du test (le test n'en a pas) ; tout target encoding **out-of-fold** ; aucun ajustement post-hoc qui ne soit pas validé en CV ; pas d'utilisation du sample_submission pour deviner la distribution.

---

## Phase 0 — Stabiliser la base (Jours 1-7)

| Jour | Action | Livrable |
|---|---|---|
| 1 | Geler une **baseline reproductible** : seed fixé partout, requirements pinné, hash des CSV, `make baseline`. | `baseline_score.json` |
| 2 | Implémenter **CV temporelle rolling** : 5 folds avec cutoffs glissants 75, 81, 87, 93, 99 ; rapporter mean ± std. | `cv_rolling.py` |
| 3 | Ajouter **CV group-aware K=5** sur `origin_account` en parallèle de la temporelle. | `cv_group.py` |
| 4 | **Target encoding OOF** : refaire `fit_target_encoding` avec KFold(5) pour produire un TE non-fuyant côté train. | Patch `features.py` |
| 5 | Retirer/garder `apply_operation_prior` selon **gain en CV**, pas selon le score de la dernière soumission. | Décision documentée |
| 6 | Mettre en place un **tracker MLflow** local (run, params, métriques, artefacts). | `mlruns/` |
| 7 | **Audit fuite** : `permutation_importance` + `target_permutation` — toute feature à importance > random sous label shuffle est suspecte. | `leak_audit.md` |

**Critère de sortie** : AP CV temporel stable ± 0.005, écart CV/LB < 0.02.

---

## Phase 1 — Feature engineering avancé (Jours 8-30)

### A. Features graphe (Jours 8-15)
Le dataset **est** un graphe biparti `origin → destination`. C'est le plus gros gisement de gain.

- J8-9 : construire le graphe avec `networkx` / `igraph` sur `train+test` (features non supervisées autorisées) ; calculer **in-degree, out-degree, PageRank, hub/authority**.
- J10-11 : **détection de cycles courts** (longueur 2-4) → fort signal de blanchiment "smurfing" et "ring".
- J12 : **fan-out / fan-in** : nb destinations distinctes / 24 périodes, nb origines distinctes recevant.
- J13 : **communautés Louvain** sur le graphe agrégé → ajouter `origin_community`, `dest_community`, `same_community`.
- J14 : **node2vec** (32 dim) par compte → embeddings concaténés au tabulaire.
- J15 : retrain LGBM seul, mesurer Δ AP, garder uniquement les features avec gain > 0.001.

### B. Features temporelles fines (Jours 16-22)
- J16 : **rolling windows** par compte sur `period` : last-1, last-5, last-20 → `amount_mean/std/max`, `nb_distinct_dest`, `nb_op_types`.
- J17 : **inter-event time** : `period - period_prev` par compte, stats glissantes ; détection de **bursts** (>3 tx en <2 périodes).
- J18 : **vélocité monétaire** : somme `amount` dans fenêtre / capacité du compte.
- J19 : **features de séquence** : ratio amount vs médiane historique du compte, **z-score** dynamique.
- J20 : **alignement balance** raffiné : flag si `origin_balance_after == 0` après tx, écart relatif `(expected - actual)/amount`.
- J21 : **patterns d'opération** : last_op == op_03 ? séquence d'op sur last-5 (trigramme encoded).
- J22 : ablation → garder uniquement les features avec gain.

### C. Features account-level (Jours 23-28)
- J23 : **profil compte** sur train+test (autorisé, pas de target) : âge (period_max - period_min), volume total, type dominant.
- J24 : **deviation features** : tx courante vs profil compte (Mahalanobis sur amount/balance).
- J25 : **bi-account features** : couples (origin, dest) → freq pair, age pair, montant cumulé pair.
- J26 : **encodage par fréquence** des comptes (frequency encoding stable train+test).
- J27 : **isolation forest score** par compte → 1 feature anomaly.
- J28 : ablation finale, sélection top-K via SHAP.

### D. Embeddings non supervisés (Jours 29-30)
- J29 : **autoencoder** sur features numériques → reconstruction error en feature.
- J30 : ajout des residuals de reconstruction par opération.

**Critère de sortie** : AP CV temporel ≥ 0.45 avec un seul LGBM.

---

## Phase 2 — Modélisation diversifiée (Jours 31-55)

L'ensemble actuel échoue parce que les 3 GBDT voient la même chose. Diversifier la **classe de modèle**.

### A. GBDT bien tunés (Jours 31-38)
- J31-32 : **Optuna** sur LightGBM, 200 essais, TPESampler, pruning Hyperband, search space large (num_leaves 16-512, mcs 5-200, lr 0.01-0.1, reg_alpha/lambda 0-10, scale_pos_weight 1-15, max_bin 63-511).
- J33-34 : idem CatBoost (depth 4-10, l2 1-20, border_count 32-254, bagging_temp 0-1).
- J35-36 : idem XGBoost (+ `dart` booster comme variante).
- J37 : **LightGBM + focal loss** (objective custom, γ=2, α=0.25) — souvent +1-2 pts en PR-AUC.
- J38 : retrain les 3 GBDT tunés, log scores OOF par fold.

### B. Modèles complémentaires (Jours 39-48)
- J39-40 : **FT-Transformer** (rtdl) sur features numériques + catégorielles, 100 epochs avec early stopping. Diversifie fortement.
- J41-42 : **TabNet** (pytorch-tabnet). Plus lent, mais attention interprétable et corrélation faible avec GBDT.
- J43 : **TabPFN-v2** sur sous-échantillon stratifié 10k (ou variante pour gros datasets si disponible) — bon prior pour blender.
- J44-45 : **Transformer séquentiel par compte** : pour chaque tx, contexte = 32 dernières tx du même compte → embedding `[CLS]` → tête binaire. Diversification forte.
- J46-47 : **réseau bipartite GNN** (GraphSAGE / GAT) sur le graphe biparti — propagation 2 hops, fenêtre temporelle glissante.
- J48 : audit corrélation des prédictions OOF (matrice de Pearson) ; viser des corrélations < 0.85 entre familles.

### C. Stacking & blending (Jours 49-55)
- J49-50 : générer prédictions OOF de tous les modèles (StratifiedGroupKFold).
- J51 : **méta-modèle niveau 2** : LightGBM léger (depth=4) sur les 7-10 prédictions OOF.
- J52 : variante méta : **logistic regression L2** sur logit des prédictions (souvent plus robuste).
- J53 : **rank averaging** vs **probability averaging** vs stacking : choisir selon CV.
- J54 : **blending pondéré** optimisé par Nelder-Mead sur l'AP CV (poids contraints positifs, somme=1).
- J55 : décider final blend vs stacking selon stabilité (std des plis).

**Critère de sortie** : AP CV ≥ 0.52 stable sur 5 plis temporels + 5 plis group.

---

## Phase 3 — Régimes avancés (Jours 56-75)

### A. Semi-supervisé (Jours 56-63)
- J56-58 : **pseudo-labels** : prédire le test, garder les exemples > 0.95 et < 0.005, réentraîner — itérer 2-3 fois en surveillant la divergence CV/LB.
- J59-60 : **noisy student** : ajout de bruit (dropout features, mixup) au réentraînement pseudo.
- J61-63 : **contrastive learning** auto-supervisé (SCARF) sur train+test → embeddings figés → feature pour GBDT.

### B. Adversarial validation & domain adaptation (Jours 64-68)
- J64 : entraîner un classifieur `is_test` ; identifier features avec drift fort (AUC > 0.7) → corriger ou supprimer.
- J65-66 : **importance weighting** : pondérer train par P(test)/P(train) estimé par le classifieur adverse.
- J67-68 : retrain avec sample weights, mesurer Δ AP.

### C. Calibration & post-processing validé en CV (Jours 69-72)
- J69 : **isotonic regression** par opération (op_01..op_05) pour caler les probas — peut aider l'AP en lissant des plateaux.
- J70 : **prior par compte** : si compte vu en train avec 0 fraude / >50 tx → léger downweight (à calibrer en CV).
- J71-72 : remplacer le `apply_operation_prior` actuel par une version apprise.

### D. Robustesse (Jours 73-75)
- J73 : multi-seed (10 seeds) sur le top blend, mesurer écart-type ; soumission = moyenne.
- J74 : **ablation finale** : retirer chaque famille de feature, mesurer perte. Documenter.
- J75 : **stress tests** : retirer le dernier mois du train, vérifier que CV reste cohérente.

**Critère de sortie** : AP CV ≥ 0.55, écart-type seeds < 0.003.

---

## Phase 4 — Optimisation finale et soumission (Jours 76-100)

- J76-80 : **hyperparameter tuning de second tour** sur features finales (Optuna 300 essais par modèle).
- J81-83 : **expérimentation focal loss vs scale_pos_weight vs no rebalancing** par modèle.
- J84-86 : **fold-wise model selection** : choisir par fold le meilleur hyperparam puis moyenner — souvent +0.005.
- J87-89 : **distillation** : entraîner un GBDT large sur les probas du meilleur stack — feature de robustesse.
- J90-92 : **CV finale "as-test"** : 5 derniers plis temporels simulant le test, ajustement final.
- J93-95 : **soumission test** sur LB (si autorisé) : 2 soumissions par jour, A/B sur composantes.
- J96-98 : **gel du pipeline**, code review, tests unitaires sur les fonctions critiques, reproducibilité totale (`make all` reproduit la soumission au bit près).
- J99 : **soumission finale** : meilleur blend ± multi-seed.
- J100 : **post-mortem** + écriture du rapport méthodologique.

---

## Hiérarchie des gains attendus (estimation conservatrice)

| Action | Δ AP attendu |
|---|---|
| Target encoding OOF correct | +0.01 à +0.02 |
| Features graphe (PageRank, cycles, communautés) | **+0.04 à +0.08** |
| Rolling windows + inter-event | +0.02 à +0.04 |
| Optuna sur GBDT | +0.01 à +0.03 |
| Diversification (GNN + Transformer séquentiel) | **+0.03 à +0.06** |
| Stacking propre | +0.01 à +0.02 |
| Pseudo-labels + adversarial | +0.01 à +0.02 |
| Multi-seed | +0.002 à +0.005 |
| **Total réaliste** | **0.36 → 0.50-0.58** |

---

## Règles d'or (anti-triche)

1. **Aucun feature** ne doit utiliser `fraud_flag` autrement que par target encoding **OOF strict**.
2. Toute feature calculée sur `train+test` doit être **non supervisée** uniquement (counts, embeddings non supervisés, graphe structurel).
3. **Aucun ajustement post-hoc** sans gain prouvé en CV temporelle ET CV group-aware.
4. **Aucune utilisation du sample_submission** pour deviner la distribution cible.
5. **Toute soumission** doit être reproductible depuis le commit Git.
6. **CV/LB doivent suivre** : si LB monte mais CV pas, c'est de l'overfit LB → rejet.

---

## Stack technique recommandée

- Tracking : MLflow local
- Hyperparam : Optuna 4.x + Hyperband pruner
- Graphe : networkx + igraph + node2vec
- GNN : PyTorch Geometric
- Tabulaire NN : rtdl-revisiting-models (FT-Transformer), pytorch-tabnet
- TabPFN : tabpfn v2
- Sequence model : PyTorch Transformer custom
- CI : pytest sur les fonctions de feature engineering (pas de fuite, idempotence)
