# src/training/detect_drift.py
"""
Wrapper de rétrocompatibilité redirigeant vers le module officiel src/audit/detect_drift.py.
"""

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.audit.detect_drift import main

if __name__ == "__main__":
    main()
