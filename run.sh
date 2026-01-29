#!/bin/bash
# run.sh - Linux/Mac startup script

echo "🔥 Log Anomaly Detection System"
echo "================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create directories
echo "Creating directories..."
mkdir -p data models/trained logs

# Check if model exists
if [ ! -f "models/trained/isolation_forest.joblib" ]; then
    echo ""
    echo "⚠️  No trained model found!"
    echo "Do you want to train the model now? (y/n)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo "Training model..."
        python training/train.py --data data/sample_logs.txt
    else
        echo "Skipping training. You can train later with:"
        echo "  python training/train.py --data data/sample_logs.txt"
    fi
fi

# Start API server
echo ""
echo "🚀 Starting API server..."
echo "Documentation: http://localhost:8000/docs"
echo "Health check: http://localhost:8000/api/v1/health"
echo ""
python app/main.py
