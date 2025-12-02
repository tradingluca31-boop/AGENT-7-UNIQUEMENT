# -*- coding: utf-8 -*-
"""
================================================================================
🤖 L'AGENT 7 PARLE - Interview Automatique
================================================================================

L'agent répond AUTOMATIQUEMENT à toutes les questions.
Pas besoin d'interaction - il s'explique tout seul!

================================================================================
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter
import time
import warnings
warnings.filterwarnings('ignore')

# Add paths
project_root = Path("C:/Users/lbye3/Desktop/GoldRL")
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src'))
sys.path.append(str(project_root / 'AGENT_V2'))
sys.path.append(str(parent_dir))

import config
from data_loader import DataLoader
from feature_engineering import FeatureEngineering
from environment.trading_env_v2_ultimate import GoldTradingEnv

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

def slow_print(text, delay=0.02):
    """Print text character by character for effect"""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()

def agent_says(text):
    """Format agent speech"""
    print(f"\n🤖 Agent 7: {text}")

def interviewer_asks(text):
    """Format interviewer question"""
    print(f"\n👤 Toi: {text}")

print("="*70)
print("🎤 INTERVIEW AUTOMATIQUE - AGENT 7 V2.1 PARLE")
print("="*70)
print("\nChargement de l'agent... (30 secondes)")

# ============================================================================
# LOAD
# ============================================================================
loader = DataLoader(verbose=False)
aligned_df, auxiliary_data = loader.load_all_data()
fe = FeatureEngineering(aligned_df, auxiliary_data, verbose=False)
features = fe.compute_all_features(config.TEST_START_DATE, config.TEST_END_DATE)
prices = auxiliary_data['xauusd_raw']['H1'].loc[features.index, ['open', 'high', 'low', 'close']]

env = GoldTradingEnv(
    features_df=features,
    prices_df=prices,
    initial_balance=100_000,
    action_space_type='discrete',
    verbose=False
)
env = Monitor(env)
vec_env = DummyVecEnv([lambda: env])

model_paths = [
    parent_dir / "models" / "checkpoints" / "agent7_critic_boost_lstm_150000_steps.zip",
    parent_dir / "models" / "best_model.zip",
]
model = None
for path in model_paths:
    if path.exists():
        model = RecurrentPPO.load(path, env=vec_env)
        break

def get_unwrapped_env(vec_env):
    env = vec_env.envs[0]
    while hasattr(env, 'env'):
        env = env.env
    return env

obs = vec_env.reset()
lstm_states = None
raw_env = get_unwrapped_env(vec_env)

print("\n✅ Agent chargé! L'interview commence...\n")
time.sleep(1)

# ============================================================================
# INTERVIEW AUTOMATIQUE
# ============================================================================

print("="*70)
print("🎤 DÉBUT DE L'INTERVIEW")
print("="*70)

# Question 1
interviewer_asks("Agent 7, quelle action vas-tu prendre maintenant?")
time.sleep(0.5)

# Get probabilities
action_counts = Counter()
for _ in range(50):
    action, _ = model.predict(obs, state=lstm_states, deterministic=False)
    action_counts[int(action[0])] += 1
probs = {k: v/50 for k, v in action_counts.items()}

# Deterministic action
action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
action_int = int(action[0])
action_names = {0: "SELL", 1: "HOLD", 2: "BUY"}

agent_says(f"Je choisis {action_names[action_int]}.")
print(f"\n   Mes probabilités:")
print(f"   - SELL: {probs.get(0,0)*100:.0f}%")
print(f"   - HOLD: {probs.get(1,0)*100:.0f}%")
print(f"   - BUY:  {probs.get(2,0)*100:.0f}%")

time.sleep(1)

# Question 2
interviewer_asks("Pourquoi tu choisis cette action?")
time.sleep(0.5)

if probs.get(1,0) > 0.7:
    agent_says("Honnêtement? J'ai peur de perdre de l'argent.")
    print("\n   Je préfère HOLD parce que:")
    print("   - Si je ne fais rien, je ne perds pas")
    print("   - BUY et SELL sont risqués")
    print("   - J'ai appris que HOLD = sécurité")
elif probs.get(1,0) > 0.5:
    agent_says("Je suis prudent. J'attends une meilleure opportunité.")
else:
    agent_says("Je vois une opportunité de trading!")

time.sleep(1)

# Question 3
interviewer_asks("As-tu peur de perdre de l'argent?")
time.sleep(0.5)

if probs.get(1,0) > 0.8:
    agent_says("OUI, j'ai très peur! 😰")
    print("\n   Je choisis HOLD", f"{probs.get(1,0)*100:.0f}% du temps.")
    print("   Mon raisonnement:")
    print("   'Si je ne trade pas, je ne peux pas perdre'")
    print("   'Le marché est dangereux, mieux vaut attendre'")
elif probs.get(1,0) > 0.5:
    agent_says("Un peu... Je suis prudent.")
else:
    agent_says("Non, je suis confiant! 😎")

time.sleep(1)

# Question 4
interviewer_asks("Quelle est ta position actuelle?")
time.sleep(0.5)

pos = raw_env.position_side
pos_names = {-1: "SHORT (vente)", 0: "FLAT (aucune)", 1: "LONG (achat)"}
agent_says(f"Ma position: {pos_names[pos]}")
print(f"\n   Balance: ${raw_env.balance:,.2f}")
print(f"   Drawdown: {raw_env.max_drawdown*100:.2f}%")
print(f"   Trades effectués: {len(raw_env.trades)}")

time.sleep(1)

# Question 5
interviewer_asks("Combien de trades as-tu fait?")
time.sleep(0.5)

n_trades = len(raw_env.trades)
if n_trades == 0:
    agent_says("Aucun trade! 😕")
    print("\n   Je n'ai ouvert aucune position.")
    print("   C'est peut-être parce que j'ai peur de perdre...")
else:
    agent_says(f"J'ai fait {n_trades} trades.")

time.sleep(1)

# Question 6: Test sur 100 steps
interviewer_asks("Montre-moi ce que tu fais sur 100 steps.")
time.sleep(0.5)

agent_says("D'accord, je vais te montrer...")

obs = vec_env.reset()
lstm_states = None
actions_100 = []

print(f"\n   Mes actions sur 100 steps:")
for step in range(100):
    action, lstm_states = model.predict(obs, state=lstm_states, deterministic=False)
    action_int = int(action[0])
    actions_100.append(action_int)
    obs, _, done, _ = vec_env.step(action)
    if done[0]:
        obs = vec_env.reset()
        lstm_states = None

counts = Counter(actions_100)
print(f"   - SELL: {counts.get(0,0)} fois ({counts.get(0,0)}%)")
print(f"   - HOLD: {counts.get(1,0)} fois ({counts.get(1,0)}%)")
print(f"   - BUY:  {counts.get(2,0)} fois ({counts.get(2,0)}%)")

time.sleep(1)

# Question 7
interviewer_asks("Pourquoi tu ne trades presque pas?")
time.sleep(0.5)

hold_pct = counts.get(1,0)
if hold_pct > 80:
    agent_says("Parce que j'ai appris que HOLD est la meilleure option! 😰")
    print("\n   Voici mon raisonnement:")
    print("   1. Quand je trade, parfois je perds de l'argent")
    print("   2. Quand je fais HOLD, je ne perds rien")
    print("   3. Donc HOLD = meilleure option (pour moi)")
    print("")
    print("   Le problème:")
    print("   - Mon reward pour HOLD ≈ 0 (neutre)")
    print("   - Mon reward pour TRADE = risque de négatif")
    print("   - J'ai appris à éviter le risque")
elif hold_pct > 50:
    agent_says("Je suis prudent, j'attends les bonnes opportunités.")
else:
    agent_says("Je trade quand je vois une opportunité!")

time.sleep(1)

# Question 8
interviewer_asks("Comment peut-on t'aider à trader plus?")
time.sleep(0.5)

agent_says("Il faudrait changer mon système de récompense!")
print("""
   Ce qui me ferait trader plus:

   1. 🎁 ME RÉCOMPENSER QUAND JE TRADE
      → Ajouter +0.03 bonus pour chaque BUY ou SELL
      → Comme ça, E[reward|TRADE] > E[reward|HOLD]

   2. 😤 ME PUNIR QUAND JE NE FAIS RIEN
      → Ajouter -0.02 penalty si je fais trop de HOLD
      → Comme ça, HOLD devient moins attractif

   3. 🎲 ME FORCER À EXPLORER PLUS
      → Garder mon entropy coefficient haut (0.25)
      → Ne pas descendre en dessous de 0.12

   4. 🧠 ME RÉ-ENTRAÎNER AVEC CES NOUVEAUX PARAMÈTRES
      → Les vieux modèles ont appris les mauvaises habitudes
      → Il faut recommencer from scratch avec les fixes
""")

time.sleep(1)

# Conclusion
print("\n" + "="*70)
print("📋 RÉSUMÉ DE L'INTERVIEW")
print("="*70)

agent_says("Voici mon auto-diagnostic:")
print(f"""
   🔍 CE QUE J'AI RÉVÉLÉ:

   - Distribution: {counts.get(1,0)}% HOLD ({"TROP!" if counts.get(1,0) > 80 else "OK"})
   - Trades ouverts: {n_trades}
   - Peur de perdre: {"OUI 😰" if probs.get(1,0) > 0.7 else "Non 😎"}
   - Cause: J'ai appris que HOLD = safe

   🛠️ SOLUTIONS DÉJÀ IMPLÉMENTÉES (V2):

   ✅ Entropy étendue: 0-50% à 0.25
   ✅ Entropy minimum: 0.12 (pas 0.05)
   ✅ Bonus trading: +0.03 pour BUY/SELL
   ✅ Penalty HOLD: -0.02 si >30 consécutifs
   ✅ Penalty collapse: -0.03 si >90% same action

   🚀 PROCHAINE ÉTAPE:

   Relancer mon entraînement avec ces nouveaux paramètres!

   cd "C:\\Users\\lbye3\\Desktop\\GoldRL\\AGENT\\AGENT 7\\ENTRAINEMENT\\FICHIER IMPORTANT AGENT 7\\launchers"
   LAUNCH_TRAINING_500K.bat
""")

print("="*70)
print("🎤 FIN DE L'INTERVIEW")
print("="*70)

vec_env.close()
