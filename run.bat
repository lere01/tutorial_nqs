@echo off

REM Check if the virtual environment already exists
IF NOT EXIST "venv\Scripts\activate" (
    echo Creating virtual environment...
    python -m venv venv
    echo Updating pip...
    venv\Scripts\python -m pip install -U pip
    echo Installing requirements...
    venv\Scripts\pip install -r requirements.txt
) ELSE (
    echo Virtual environment already exists. Skipping creation.
)

echo Running the application...
venv\Scripts\streamlit run app.py
