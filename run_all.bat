@echo off
REM Runs the whole offline pipeline in order. Run from the repo root with the venv active.
python scripts\eda.py || exit /b 1
python scripts\build_features.py || exit /b 1
python scripts\train_models.py || exit /b 1
python scripts\tune_optuna.py || exit /b 1
python scripts\eval_hmm.py || exit /b 1
python scripts\make_maps.py student_test 00000052 || exit /b 1
echo ALL DONE
