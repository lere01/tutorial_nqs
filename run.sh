#!/bin/bash

# Check if the virtual environment already exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Updating pip..."
    ./venv/bin/python -m pip install -U pip
    echo "Installing requirements..."
    ./venv/bin/pip install -r requirements.txt
else
    echo "Virtual environment already exists. Skipping creation."
fi

echo "Running the application..."
./venv/bin/streamlit run app.py
