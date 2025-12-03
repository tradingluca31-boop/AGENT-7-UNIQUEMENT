# 🔄 MASTER WORKFLOW GUIDE - Updates Tracking System

> **OBJECTIF** : Système de suivi des mises à jour/améliorations réutilisable pour tous les agents

---

## 📂 STRUCTURE DES UPDATES

```
updates/
  ├── MASTER_WORKFLOW_GUIDE.md           ← Ce fichier (copier pour autres agents)
  └── YYYY-MM-DD-description-courte/     ← Dossier daté pour chaque update
      ├── DESCRIPTION.md                  ← OBLIGATOIRE - Documentation complète
      ├── RESULTS.txt                     ← Résultats training/backtest
      ├── fichiers_modifiés.py            ← Code modifié (si applicable)
      ├── BENCHMARK.csv                   ← Métriques comparatives
      └── captures/                       ← Screenshots/graphiques
```

---

## 🎯 NAMING CONVENTION (CRITIQUE)

### Format Dossier Update
```
YYYY-MM-DD-description-courte

Exemples :
  ✅ 2025-12-03-dataset-2008-training-500k
  ✅ 2025-12-05-adaptive-entropy-fix
  ✅ 2025-12-10-curriculum-v2-test
  ✅ 2026-01-15-meta-agent-integration

  ❌ update1                (pas de date)
  ❌ new-feature            (pas de date)
  ❌ 12-03-fix              (format date incomplet)
```

**Pourquoi cette convention ?**
- ✅ Tri chronologique automatique
- ✅ Description claire visible immédiatement
- ✅ Pas de conflits de noms
- ✅ Facile à retrouver dans l'historique

---

## 📝 DESCRIPTION.md TEMPLATE (OBLIGATOIRE)

Chaque dossier update **DOIT** contenir un `DESCRIPTION.md` avec ce format :

```markdown
# [Titre de l'Update] - YYYY-MM-DD

## 🎯 Objectif
Pourquoi cette modification ? Quel problème résout-elle ?

## 🔧 Changements Techniques

### Fichiers Modifiés
- `chemin/fichier1.py` : Description changement
- `config.py` : Paramètres modifiés (ligne X)
- etc.

### Paramètres Changés
| Paramètre | Avant | Après | Raison |
|-----------|-------|-------|--------|
| TRAIN_START_DATE | 2015-01-01 | 2008-01-01 | +7 ans de données |
| total_timesteps | 1,500,000 | 500,000 | Test incrémental |

## 📊 Résultats

### Training Stats
- **Durée** : X heures
- **Steps** : X
- **Dataset** : YYYY-YYYY (X bars)

### Performance Metrics
| Métrique | Valeur | Benchmark | Amélioration |
|----------|--------|-----------|--------------|
| ROI % | X | Y | +Z% |
| Sharpe | X | Y | +Z |
| Win Rate % | X | Y | +Z% |
| Profit Factor | X | Y | +Z |
| Max DD % | X | Y | -Z% |

### Best Checkpoint
- **Step** : X
- **Score** : Y/10
- **ROI** : Z%

## 🔍 Analyse

### Points Positifs ✅
1. ...
2. ...

### Points Négatifs ❌
1. ...
2. ...

### Observations
- ...

## 🚀 Next Steps

### Recommandations
1. [ ] Action 1
2. [ ] Action 2
3. [ ] Action 3

### Hypothèses à Tester
- ...

---

**Auteur** : Claude + User
**Date** : YYYY-MM-DD
**Agent** : Agent X
**Version** : VX.X
```

---

## 🔄 WORKFLOW COMPLET (STEP-BY-STEP)

### 1️⃣ Avant Modification
```bash
# Créer dossier update avec date du jour
cd "C:\Users\lbye3\Desktop\GoldRL\AGENT\AGENT 7\ENTRAINEMENT\FICHIER IMPORTANT AGENT 7"
mkdir updates/$(date +%Y-%m-%d)-description-courte

# Exemple Windows
mkdir updates\2025-12-03-mon-update
```

### 2️⃣ Pendant Modification
- ✍️ Noter TOUS les fichiers modifiés
- 📸 Capturer paramètres AVANT/APRÈS
- 💾 Sauvegarder versions originales si grosse modification

### 3️⃣ Après Training/Test
```bash
# Copier résultats dans dossier update
cp output/training_log.txt updates/2025-12-03-mon-update/RESULTS.txt
cp output/metrics.csv updates/2025-12-03-mon-update/BENCHMARK.csv
```

### 4️⃣ Documentation
```bash
# Créer DESCRIPTION.md avec template ci-dessus
# Remplir TOUTES les sections
# Être précis sur les chiffres
```

### 5️⃣ Git Commit & Push (AUTOMATIQUE)
```bash
cd "C:\Users\lbye3\Desktop\GoldRL\AGENT\AGENT 7\ENTRAINEMENT\FICHIER IMPORTANT AGENT 7"

# Add update folder
git add updates/2025-12-03-mon-update/

# Commit avec message clair
git commit -m "feat: [Description courte de l'update]

- Changement 1
- Changement 2
- Résultats : ROI X%, Sharpe Y

Closes #issue-number (si applicable)"

# Push to GitHub
git push origin main
```

---

## 📋 CHECKLIST VALIDATION UPDATE

Avant de considérer un update comme "complet", vérifier :

- [ ] Dossier nommé avec date ISO (YYYY-MM-DD-description)
- [ ] `DESCRIPTION.md` existe et complet (tous les champs remplis)
- [ ] Résultats training inclus (RESULTS.txt ou équivalent)
- [ ] Tableau comparatif AVANT/APRÈS (si amélioration)
- [ ] Fichiers modifiés documentés avec lignes précises
- [ ] Next steps identifiés
- [ ] Git commit créé avec message clair
- [ ] Push vers GitHub effectué
- [ ] Aucune donnée sensible committée (credentials, API keys)

---

## 🎓 EXEMPLES D'UPDATES TYPES

### Type 1 : Dataset Upgrade
```
2025-12-03-dataset-2008-training-500k/
  ├── DESCRIPTION.md          (doc complète upgrade 2015→2008)
  ├── RESULTS.txt             (logs training 500K)
  ├── BENCHMARK.csv           (comparaison checkpoints)
  └── config_changes.txt      (TRAIN_START_DATE modifié)
```

### Type 2 : Hyperparameter Tuning
```
2025-12-05-learning-rate-decay-test/
  ├── DESCRIPTION.md          (test LR 1e-5 vs 5e-6)
  ├── RESULTS.txt             (metrics avec nouveau LR)
  ├── optuna_results.csv      (si Optuna utilisé)
  └── tensorboard_screenshot.png
```

### Type 3 : Architecture Change
```
2025-12-10-add-lstm-layer/
  ├── DESCRIPTION.md          (ajout LSTM 256 units)
  ├── model_architecture.txt  (before/after)
  ├── training_curves.png
  └── modified_files/
      └── train_from_scratch.py
```

### Type 4 : Bug Fix
```
2025-12-15-fix-mode-collapse/
  ├── DESCRIPTION.md          (fix adaptive entropy)
  ├── BEFORE_AFTER.md         (diversity score 0.1 → 0.8)
  ├── diagnostic_output.txt
  └── entropy_schedule.png
```

---

## 🔁 RÉUTILISATION POUR AUTRES AGENTS

Pour utiliser ce workflow sur **Agent 8, 9, 11, Meta-Agent** :

1. **Copier ce fichier** (`MASTER_WORKFLOW_GUIDE.md`) dans le dossier agent
   ```bash
   cp "C:\Users\lbye3\Desktop\GoldRL\AGENT\AGENT 7\ENTRAINEMENT\FICHIER IMPORTANT AGENT 7\updates\MASTER_WORKFLOW_GUIDE.md" \
      "C:\Users\lbye3\Desktop\GoldRL\AGENT\AGENT 8\ENTRAINEMENT\updates\"
   ```

2. **Créer dossier updates/** dans l'agent cible
   ```bash
   mkdir "C:\Users\lbye3\Desktop\GoldRL\AGENT\AGENT 8\ENTRAINEMENT\updates"
   ```

3. **Suivre le même workflow** (création dossiers datés, DESCRIPTION.md, etc.)

4. **Adapter les métriques** selon l'algorithme :
   - PPO (Agent 7) : entropy_coef, clip_range
   - SAC (Agent 8) : ent_coef='auto', tau
   - TD3 (Agent 9) : policy_delay, target_noise
   - A2C (Agent 11) : n_steps, vf_coef

---

## 🚨 ERREURS FRÉQUENTES À ÉVITER

❌ **Oublier la date dans le nom du dossier**
   → Solution : Toujours format YYYY-MM-DD au début

❌ **DESCRIPTION.md vide ou incomplet**
   → Solution : Utiliser template complet ci-dessus

❌ **Pas de métriques AVANT/APRÈS**
   → Solution : Toujours benchmarker vs version précédente

❌ **Commit "WIP" ou "test" sans description**
   → Solution : Messages de commit descriptifs

❌ **Mélanger plusieurs updates dans un dossier**
   → Solution : 1 update = 1 dossier daté

❌ **Pas de push GitHub**
   → Solution : Toujours `git push` après commit

❌ **Copier-coller DESCRIPTION.md sans modifier**
   → Solution : Personnaliser pour chaque update

---

## 📊 TRACKING LONG TERME

### Créer un CHANGELOG.md (optionnel mais recommandé)
```markdown
# CHANGELOG - Agent X

## 2025-12-10 - Add LSTM Layer
- ROI: 12% → 15% (+25%)
- See: updates/2025-12-10-add-lstm-layer/

## 2025-12-05 - Learning Rate Decay
- Sharpe: 1.2 → 1.4 (+17%)
- See: updates/2025-12-05-learning-rate-decay-test/

## 2025-12-03 - Dataset Upgrade 2008
- Bars: 35K → 75K (+114%)
- See: updates/2025-12-03-dataset-2008-training-500k/
```

---

## 🎯 OBJECTIF FINAL

**Ce workflow permet de** :
- ✅ Tracer TOUS les changements chronologiquement
- ✅ Comparer performances entre versions
- ✅ Reproduire n'importe quelle expérience
- ✅ Partager updates avec collaborateurs (GitHub)
- ✅ Éviter "pourquoi ça marchait avant ?"
- ✅ Documenter apprentissages pour futures décisions

**Standard** : Hedge fund / Trading institutionnel - Documentation rigoureuse obligatoire

---

**🏆 Best Practice** : "Si ce n'est pas documenté, ça n'est jamais arrivé."

---

*Version: 1.0*
*Créé: 2025-12-03*
*Agent: 7 (PPO + LSTM)*
*Réutilisable: Tous agents*
