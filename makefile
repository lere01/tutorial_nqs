# Define variables
VENV = venv
PYTHON = python3
PIP = $(VENV)/bin/pip
STREAMLIT = $(VENV)/bin/streamlit

# Windows specific variables
ifdef OS
	ifeq ($(OS),Windows_NT)
	PYTHON = python
	PIP = $(VENV)/Scripts/pip
	STREAMLIT = $(VENV)/Scripts/streamlit
	endif
endif

# Default target
all: run

# Create virtual environment
$(VENV)/bin/activate: requirements.txt
ifeq ($(OS),Windows_NT)
	if not exist $(VENV)\Scripts\activate (
	$(PYTHON) -m venv $(VENV) && \
	$(PIP) install -U pip && \
	$(PIP) install -r requirements.txt
	)
else
	test -d $(VENV) || ( \
	$(PYTHON) -m venv $(VENV) && \
	$(PIP) install -U pip && \
	$(PIP) install -r requirements.txt \
	)
endif

# Run the application
run: $(VENV)/bin/activate
ifeq ($(OS),Windows_NT)
	@echo "Running the application..."
	$(STREAMLIT) run app.py
else
	@echo "Running the application..."
	$(STREAMLIT) run app.py
endif

# Stop the application
stop:
ifeq ($(OS),Windows_NT)
	@echo "Stopping the application..."
	taskkill /IM "streamlit.exe" /F
else
	@echo "Stopping the application..."
	pkill -f "streamlit run app.py"
endif

# Clean up the virtual environment
clean:
	@echo "Cleaning up environment..."
	rm -rf $(VENV)
