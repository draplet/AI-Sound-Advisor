# conftest.py
"""
Root-level pytest configuration.

Inserts the project root directory onto sys.path so that
`from src.module import ...` resolves correctly regardless of
how pytest is invoked.
"""
import os
import sys

# Always ensure the directory that *contains* src/ is on the path.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))