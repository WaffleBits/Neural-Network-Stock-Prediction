@echo off
call venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python stock_predictor.py
pause