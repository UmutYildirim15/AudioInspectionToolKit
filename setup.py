import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required_modules = [
    "numpy", "pandas", "PyQt5", "matplotlib", "reportlab", "librosa", "pydub", "mediainfo"
]

import_names = {
    "numpy": "np",
    "pandas": "pd",
    "PyQt5.QtGui": "QPixmap, QIcon",
    "PyQt5.QtWidgets": "(QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, QVBoxLayout, QWidget, QListWidget, QProgressBar, QTextEdit, QComboBox, QLineEdit, QHBoxLayout)",
    "PyQt5.QtCore": "Qt",
    "matplotlib.pyplot": "plt",
    "reportlab.lib.pagesizes": "letter",
    "reportlab.pdfgen": "canvas",
    "librosa": "librosa",
    "pydub.utils": "mediainfo"
}

for module in required_modules:
    try:
        __import__(module)
    except ImportError:
        print(f"{module} could not find, installing...")
        install_package(module)

import os
import numpy as np
import pandas as pd
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QFileDialog,
                             QLabel, QVBoxLayout, QWidget, QListWidget, QProgressBar,
                             QTextEdit, QComboBox, QLineEdit, QHBoxLayout)
from PyQt5.QtCore import Qt
from matplotlib import pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import librosa
from pydub.utils import mediainfo

print("All necessary modules installed and imported successfully!")
