@echo off
echo ================================================================================
echo AGENT 7 V2.2 OPTIMAL - TRAINING 1.5M STEPS
echo ================================================================================
echo.
echo [FIX] Exponential Slow Decay: 0.30 (600K) -^> 0.18 (1.5M)
echo [FIX] Curriculum 6 Levels: 250K par level
echo [FIX] Critic Boost reduced: vf_coef = 0.6
echo [FIX] Total timesteps: 1,500,000
echo.
echo PROBLEMES V2.1 FIXES:
echo   - Entropy decay trop rapide (0.25 -^> 0.12 en 150K)
echo   - Curriculum 4 levels insuffisant
echo   - Critic Boost trop agressif (vf_coef=1.0)
echo   - Training trop court (500K pour 13 ans)
echo   - Peak 200K puis degradation
echo.
echo SOLUTIONS V2.2 OPTIMAL:
echo   - Entropy 0.30 constant pendant 600K
echo   - Decay exponentiel LENT (600K-1.2M)
echo   - Entropy finale 0.18 (3x plus haute)
echo   - 6 curriculum levels (250K each)
echo   - Critic Boost 0.6 (anti-overfitting)
echo.
echo PERFORMANCE ATTENDUE:
echo   - ROI: 25-30%% (vs 18%% V2.1)
echo   - Sharpe: 2.5-3.0 (hedge fund grade)
echo   - Win Rate: 70-75%%
echo   - Max DD: ^<6%% (FTMO compliant)
echo.
echo DUREE: ~18 heures
echo CHECKPOINTS: 30 (every 50K steps)
echo.
echo ================================================================================
pause

cd /d "%~dp0\..\training"
python train_from_scratch_OPTIMAL_1M5.py

echo.
echo ================================================================================
pause