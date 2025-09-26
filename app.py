# PeerReviewer Main Application Entry Point
# For development and quick testing

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the ULTIMATE application  
from ultimate_app import app

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=8000)
