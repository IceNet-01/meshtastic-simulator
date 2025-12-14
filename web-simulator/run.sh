#!/bin/bash
# Meshtastic Web Simulator startup script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Check if dependencies are installed
if ! python -c "import flask" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Ensure upstream meshtasticator exists
if [ ! -d "../upstream-meshtasticator" ]; then
    echo "Cloning Meshtasticator..."
    cd ..
    git clone https://github.com/meshtastic/Meshtasticator.git upstream-meshtasticator
    cd web-simulator
fi

# Create output directory
mkdir -p ../upstream-meshtasticator/out

# Run the application
echo "Starting Meshtastic Web Simulator on http://localhost:4000"
python app.py
