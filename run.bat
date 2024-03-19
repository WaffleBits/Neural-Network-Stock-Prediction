@echo off
echo Setting up Python environment...

IF NOT EXIST "../env" (
    python -m venv ../env
    echo Virtual environment created.
)

echo Activating environment...
CALL ../env/Scripts/activate

echo Installing dependencies...
pip install -r ../requirements.txt

echo Running the model...
python ../src/model.py

pause
