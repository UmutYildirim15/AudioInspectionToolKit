import os
import sys

import numpy as np
import pandas as pd
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton,
                             QFileDialog, QLabel, QVBoxLayout, QWidget, QListWidget,
                             QProgressBar, QTextEdit, QComboBox, QLineEdit, QHBoxLayout)
from PyQt5.QtCore import Qt
from matplotlib import pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import AudioFileChecker


class AudioInspectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.result_text = None
        self.add_format_button = None
        self.new_format_input = None
        self.supported_formats = ['wav', 'mp3', 'flac', 'm4a']
        self.target_rates = [44100, 48000]
        self.bit_rates = [8, 16, 24, 32]
        self.current_bit_rates = [str(bit) for bit in self.bit_rates]
        self.audio_checker = AudioFileChecker.AudioFileChecker(self.supported_formats, self.target_rates)
        self.current_analysis = None
        self.noise_levels = []
        self.snr_levels = []
        self.clipping_data = []
        self.channel_modes = []

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Audio File Inspection Toolkit')
        self.setGeometry(0, 0, 1200, 900)

        self.setWindowIcon(QIcon(self.resource_path('logo.ico')))
        logo = QLabel(self)
        pixmap = QPixmap(self.resource_path('images.png'))
        logo.setPixmap(pixmap)
        logo.setAlignment(Qt.AlignCenter)

        central_widget = QWidget()
        layout = QVBoxLayout()

        layout.addWidget(logo)
        self.label = QLabel("Drag and drop audio files here", self)
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        self.file_list = QListWidget(self)
        self.file_list.setSelectionMode(QListWidget.MultiSelection)
        self.file_list.setStyleSheet(""" 
                            QListWidget {
                                background-color: #f9f9f9;
                                border: 1px dashed #ccc;
                                border-radius: 5px;
                                padding: 10px;
                            }
                            QListWidget::item:hover {
                                background-color: #e0e0e0;
                            }
                        """)
        layout.addWidget(self.file_list)

        self.setAcceptDrops(True)

        self.current_formats_label = QLabel(
            "Current Formats: " + ", ".join(self.supported_formats) +
            " | Current Sampling Rates: " + ", ".join(f"{rate}" for rate in self.target_rates) +
            " | Current Bit Depths: " + ", ".join(f"{bit}" for bit in self.current_bit_rates)
        )

        layout.addWidget(self.current_formats_label)

        dropdown_style = """
                QComboBox {
                    background-color: #ffffff;
                    border: 1px solid #ccc;
                    padding: 5px;
                    font-size: 14px;
                }
                QComboBox:focus {
                    border-color: #007BFF;
                }
                QComboBox::drop-down {
                    border-left: 1px solid #ccc;
                    background-color: #f9f9f9;
                }
                QComboBox::down-arrow {
                    image: url(down-arrow.png);
                }
            """

        # Dropdown for analysis type
        self.analysis_type = QComboBox(self)
        self.analysis_type.setStyleSheet(dropdown_style)
        self.analysis_type.addItems([
            "Verify Format and Sampling Rate",
            "Analyze Background Noise",
            "Analyze SNR",
            "Detect Clipping",
            "Analyze Reverb",
            "Inspect Channel Mode",
            "Verify Bit Depth"
        ])
        layout.addWidget(self.analysis_type)

        # Sample rate input adjustments (Half width)
        # Target Sample Rate layout
        target_rate_layout = QHBoxLayout()

        # Target Sample Rate input
        self.target_rate_input = QLineEdit(self)
        self.target_rate_input.setPlaceholderText("Enter New Target Sample Rate (Default 48000 Hz)")
        self.target_rate_input.setFixedWidth(500)  # Half-width adjustment
        self.target_rate_input.editingFinished.connect(self.change_rate)
        self.target_rate_input.setStyleSheet(""" 
                                        QLineEdit {
                                            background-color: #f9f9f9;
                                            border: 1px solid #ccc;
                                            border-radius: 5px;
                                            padding: 5px;
                                        }
                                        QLineEdit:focus {
                                            border: 1px solid #007BFF;
                                        }
                                    """)

        # Dropdown for Current Sampling Rates
        self.sampling_rate_dropdown = QComboBox(self)
        self.sampling_rate_dropdown.setStyleSheet(dropdown_style)
        self.sampling_rate_dropdown.addItems([str(rate) for rate in self.target_rates])

        # Add Button for Sampling Rate
        self.add_rate_button = QPushButton("Add Sampling Rate", self)
        self.add_rate_button.clicked.connect(self.add_sampling_rate)
        self.add_rate_button.setStyleSheet(""" 
                                                QPushButton {
                                                    background-color: #35d7de;  
                                                    color: white;
                                                    border: none;
                                                    padding: 10px;
                                                    border-radius: 5px;
                                                }
                                                QPushButton:hover {
                                                    background-color: #8ee5e8;  
                                                }
                                            """)

        # Remove Button for Sampling Rate
        self.remove_rate_button = QPushButton("Remove", self)
        self.remove_rate_button.clicked.connect(self.remove_sampling_rate)
        self.remove_rate_button.setStyleSheet(""" 
                                                    QPushButton {
                                                        background-color: #35d7de;  
                                                        color: white;
                                                        border: none;
                                                        padding: 10px;
                                                        border-radius: 5px;
                                                    }
                                                    QPushButton:hover {
                                                        background-color: #8ee5e8;  
                                                    }
                                                """)

        # Adding widgets to the layout
        target_rate_layout.addWidget(self.sampling_rate_dropdown)
        target_rate_layout.addWidget(self.target_rate_input)
        target_rate_layout.addWidget(self.add_rate_button)
        target_rate_layout.addWidget(self.remove_rate_button)

        layout.addLayout(target_rate_layout)

        # Horizontal layout for the buttons
        button_layout = QHBoxLayout()

        # Remove button
        self.remove_button = QPushButton('Remove Selected Files', self)
        self.remove_button.clicked.connect(self.remove_selected_files)
        self.remove_button.setStyleSheet(""" 
                                QPushButton {
                                    background-color: #0a475c;
                                    color: white;
                                    border: none;
                                    padding: 10px;
                                    border-radius: 5px;
                                    width: 80px; 
                                }
                                QPushButton:hover {
                                    background-color: #011c25;
                                }
                            """)


        self.upload_files_button = QPushButton('Upload New Files', self)
        self.upload_files_button.clicked.connect(self.upload_files)
        self.upload_files_button.setStyleSheet(""" 
                                        QPushButton {
                                            background-color: #800080;
                                            color: white;
                                            border: none;
                                            padding: 10px;
                                            border-radius: 5px;
                                            width: 80px; 
                                        }
                                        QPushButton:hover {
                                            background-color: #9932CC;
                                        }
                                    """)

        # Remove All button
        self.remove_all_button = QPushButton('Remove All Files', self)
        self.remove_all_button.clicked.connect(self.remove_all_files)
        self.remove_all_button.setStyleSheet(""" 
                                        QPushButton {
                                            background-color: #011c25;
                                            color: white;
                                            border: none;
                                            padding: 10px;
                                            border-radius: 5px;
                                            width: 80px; 
                                        }
                                        QPushButton:hover {
                                            background-color: #98aeb6;
                                        }
                                    """)

        button_layout.addWidget(self.upload_files_button)
        button_layout.addWidget(self.remove_button)
        button_layout.addWidget(self.remove_all_button)

        # Format input adjustments (Half width)
        # Format layout
        format_layout = QHBoxLayout()

        # New format input
        self.new_format_input = QLineEdit(self)
        self.new_format_input.setPlaceholderText("Add New Format (e.g., 'ogg')")
        self.new_format_input.setStyleSheet(""" 
                                       QLineEdit {
                                           background-color: #f9f9f9;
                                           border: 1px solid #ccc;
                                           border-radius: 5px;
                                           padding: 5px;
                                       }
                                       QLineEdit:focus {
                                           border: 1px solid #007BFF;
                                       }
                                   """)
        self.new_format_input.setFixedWidth(500)

        # Dropdown for current formats
        self.format_dropdown = QComboBox(self)
        self.format_dropdown.setStyleSheet(dropdown_style)
        self.format_dropdown.addItems(self.supported_formats)

        # Add Button for format
        self.add_format_button = QPushButton('Add Format', self)
        self.add_format_button.clicked.connect(self.add_format)
        self.add_format_button.setStyleSheet(""" 
                                    QPushButton {
                                        background-color: #4CAF50;  
                                        color: white;
                                        border: none;
                                        padding: 10px;
                                        border-radius: 5px;
                                    }
                                    QPushButton:hover {
                                        background-color: #45a049;  
                                    }
                                """)

        # Remove Button for format
        self.remove_format_button = QPushButton('Remove Format', self)
        self.remove_format_button.setStyleSheet(""" 
                                    QPushButton {
                                        background-color: #4CAF50;  
                                        color: white;
                                        border: none;
                                        padding: 10px;
                                        border-radius: 5px;
                                    }
                                    QPushButton:hover {
                                        background-color: #45a049;  
                                    }
                                """)
        self.remove_format_button.clicked.connect(self.remove_format)

        # Adding widgets to format layout
        format_layout.addWidget(self.format_dropdown)
        format_layout.addWidget(self.new_format_input)
        format_layout.addWidget(self.add_format_button)
        format_layout.addWidget(self.remove_format_button)

        layout.addLayout(format_layout)

        # Bit rate input (unchanged)
        self.bit_rate_input = QLineEdit(self)
        self.bit_rate_input.setPlaceholderText("Add New Bit Depth")
        self.bit_rate_input.setStyleSheet(""" 
                                   QLineEdit {
                                       background-color: #f9f9f9;
                                       border: 1px solid #ccc;
                                       border-radius: 5px;
                                       padding: 5px;
                                   }
                                   QLineEdit:focus {
                                       border: 1px solid #007BFF;
                                   }
                               """)
        self.bit_rate_input.setFixedWidth(500)
        self.add_bit_rate_button = QPushButton("Add Bit Rate", self)
        self.add_bit_rate_button.setStyleSheet(""" 
                                QPushButton {
                                    background-color: #abb76c;  
                                    color: white;
                                    border: none;
                                    padding: 10px;
                                    border-radius: 5px;
                                }
                                QPushButton:hover {
                                    background-color: #c6ce9d;  
                                }
                            """)
        self.remove_bit_rate_button = QPushButton("Remove Bit Rate", self)
        self.remove_bit_rate_button.setStyleSheet(""" 
                                QPushButton {
                                    background-color: #abb76c;  
                                    color: white;
                                    border: none;
                                    padding: 10px;
                                    border-radius: 5px;
                                }
                                QPushButton:hover {
                                    background-color: #c6ce9d;  
                                }
                            """)
        self.current_bit_rates_dropdown = QComboBox(self)
        self.current_bit_rates_dropdown.setStyleSheet(dropdown_style)
        self.current_bit_rates_dropdown.addItems([str(bit) for bit in self.bit_rates])

        self.add_bit_rate_button.clicked.connect(self.add_bit_rate)
        self.remove_bit_rate_button.clicked.connect(self.remove_bit_rate)

        bit_rate_layout = QHBoxLayout()
        bit_rate_layout.addWidget(self.current_bit_rates_dropdown)
        bit_rate_layout.addWidget(self.bit_rate_input)
        bit_rate_layout.addWidget(self.add_bit_rate_button)
        bit_rate_layout.addWidget(self.remove_bit_rate_button)

        layout.addLayout(bit_rate_layout)

        # Analyze Selected button
        analyze_button = QPushButton('Analyze Selected', self)
        analyze_button.clicked.connect(self.perform_analysis)
        analyze_button.setStyleSheet(""" 
                            QPushButton {
                                background-color: #007BFF;
                                color: white;
                                border: none;
                                padding: 10px;
                                border-radius: 5px;
                            }
                            QPushButton:hover {
                                background-color: #0056b3;
                            }
                        """)
        button_layout.addWidget(analyze_button)

        # Run All Analyses button
        all_analysis_button = QPushButton('Run All Analyses', self)
        all_analysis_button.clicked.connect(self.perform_all_analyses)
        all_analysis_button.setStyleSheet(""" 
                            QPushButton {
                                background-color: #FF5733;
                                color: white;
                                border: none;
                                padding: 10px;
                                border-radius: 5px;
                            }
                            QPushButton:hover {
                                background-color: #0056b3;
                            }
                        """)
        button_layout.addWidget(all_analysis_button)

        layout.addLayout(button_layout)

        self.result_display = QTextEdit(self)
        self.result_display.setReadOnly(True)
        self.result_display.setStyleSheet(""" 
                            QTextEdit {
                                background-color: #ffffff;
                                border: 1px solid #ccc;
                                border-radius: 5px;
                                padding: 10px;
                                font-family: 'Arial';
                                font-size: 14px;
                            }
                        """)
        self.result_display.setMinimumHeight(240)  # Minimum yükseklik ayarla
        layout.addWidget(self.result_display)

        self.download_buttons_layout = QHBoxLayout()

        self.pdf_button = QPushButton('Download Results as PDF', self)
        self.pdf_button.clicked.connect(self.download_pdf)
        self.pdf_button.setStyleSheet(""" 
                            QPushButton {
                                background-color: #e74c3c;
                                color: white;
                                border: none;
                                padding: 10px;
                                border-radius: 5px;
                            }
                            QPushButton:hover {
                                background-color: #c0392b;
                            }
                        """)
        self.download_buttons_layout.addWidget(self.pdf_button)

        self.excel_button = QPushButton('Download Results as Excel', self)
        self.excel_button.clicked.connect(self.download_excel)
        self.excel_button.setStyleSheet(""" 
                            QPushButton {
                                background-color: #2ecc71;
                                color: white;
                                border: none;
                                padding: 10px;
                                border-radius: 5px;
                            }
                            QPushButton:hover {
                                background-color: #27ae60;
                            }
                        """)
        self.download_buttons_layout.addWidget(self.excel_button)

        self.csv_button = QPushButton('Download Results as CSV', self)
        self.csv_button.clicked.connect(self.download_csv)
        self.csv_button.setStyleSheet(""" 
                            QPushButton {
                                background-color: #f39c12;
                                color: white;
                                border: none;
                                padding: 10px;
                                border-radius: 5px;
                            }
                            QPushButton:hover {
                                background-color: #e67e22;
                            }
                        """)
        self.download_buttons_layout.addWidget(self.csv_button)

        self.statistics_button = QPushButton('Download Statistics', self)
        self.statistics_button.clicked.connect(self.download_statistics)  # Placeholder for the function
        self.statistics_button.setStyleSheet(""" 
                            QPushButton {
                                background-color: #3498db;
                                color: white;
                                border: none;
                                padding: 10px;
                                border-radius: 5px;
                            }
                            QPushButton:hover {
                                background-color: #2980b9;
                            }
                        """)
        self.download_buttons_layout.addWidget(self.statistics_button)

        self.showstatistics_button = QPushButton('Show Statistics', self)
        self.showstatistics_button.clicked.connect(self.show_statistics)  # Placeholder for the function
        self.showstatistics_button.setStyleSheet(""" 
                                    QPushButton {
                                        background-color: #0791C1;
                                        color: white;
                                        border: none;
                                        padding: 10px;
                                        border-radius: 5px;
                                    }
                                    QPushButton:hover {
                                        background-color: #2980b9;
                                    }
                                """)
        self.download_buttons_layout.addWidget(self.showstatistics_button)

        layout.addLayout(self.download_buttons_layout)

        # Progress Bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setStyleSheet("""
        QProgressBar {
            border: 2px solid #555;
            border-radius: 5px;
            text-align: center;
            font-size: 14px;
            background-color: #f0f0f0;
        }
        QProgressBar::chunk {
            background-color: #5cb85c;
            width: 20px;
        }
    """)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        for file in files:
            self.file_list.addItem(file)

    def update_inputs(self):
        analysis_type = self.analysis_type.currentText()
        if analysis_type == "Verify Format and Sampling Rate":
            self.target_rate_input.show()
            self.format_input.show()
            self.new_format_input.show()
            self.add_format_button.show()
        else:
            self.target_rate_input.hide()
            self.format_input.hide()
            self.new_format_input.hide()
            self.add_format_button.hide()

    def add_format(self):
        new_format = self.new_format_input.text().strip()
        if new_format and new_format not in self.supported_formats:
            self.supported_formats.append(new_format)
            self.new_format_input.clear()
            self.update_current_formats()
            self.format_dropdown.addItem(new_format)
        else:
            self.result_display.append(f"Format {new_format} is already in the list or invalid.")

    def remove_format(self):
        selected_format = self.format_dropdown.currentText()
        if selected_format:
            self.supported_formats.remove(selected_format)
            self.format_dropdown.removeItem(self.format_dropdown.currentIndex())
            self.update_current_formats()


    def add_sampling_rate(self):
        try:
            new_rate = int(self.target_rate_input.text())
            if new_rate not in self.target_rates:
                self.target_rates.append(new_rate)
                self.sampling_rate_dropdown.addItem(str(new_rate))
                self.update_current_formats()
            else:
                self.result_display.append(f"Sampling rate {new_rate} Hz is already in the list.")
        except ValueError:
            self.result_display.append("Error: Invalid sampling rate input.")

    def remove_sampling_rate(self):
        selected_rate = self.sampling_rate_dropdown.currentText()
        if selected_rate:
            self.target_rates.remove(int(selected_rate))
            self.sampling_rate_dropdown.removeItem(self.sampling_rate_dropdown.currentIndex())
            self.update_current_formats()

    def change_rate(self):
        try:
            new_rate = int(self.target_rate_input.text())
            if new_rate not in self.target_rates:
                self.target_rates.append(new_rate)
                self.target_rate_input.clear()
                self.update_current_formats()
                self.sampling_rate_dropdown.addItem(str(new_rate))
            else:
                self.result_display.append(f"Sampling rate {new_rate} Hz is already in the list.")
        except ValueError:
            self.result_display.append("Error: Invalid sampling rate input. Please enter a valid number.")

    def update_current_formats(self):
        self.current_formats_label.setText(
            "Current Formats: " + ", ".join(self.supported_formats) +
            " | Current Sampling Rates: " + ", ".join(f"{rate}" for rate in self.target_rates) +
            " | Current Bit Depths: " + ", ".join(f"{bit}" for bit in self.current_bit_rates))

    def add_bit_rate(self):
        new_bit_rate = self.bit_rate_input.text()
        if new_bit_rate.isdigit() and new_bit_rate not in self.current_bit_rates:  # Bit rate'in sayı olduğundan emin olun
            self.current_bit_rates.append(new_bit_rate)
            self.update_bit_rate_dropdown()
            self.update_current_formats()

    def remove_bit_rate(self):
        current_bit_rate = self.current_bit_rates_dropdown.currentText()
        if current_bit_rate in self.current_bit_rates:
            self.current_bit_rates.remove(current_bit_rate)
            self.update_bit_rate_dropdown()
            self.update_current_formats()

    def update_bit_rate_dropdown(self):
        self.current_bit_rates_dropdown.clear()
        self.current_bit_rates_dropdown.addItems(self.current_bit_rates)

    def upload_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, 'Upload Files')
        for file in files:
            self.file_list.addItem(file)

    def remove_all_files(self):
        self.file_list.clear()

    def perform_analysis(self):
        QApplication.processEvents()
        selected_files = [item.text() for item in self.file_list.selectedItems()] or \
                         [self.file_list.item(i).text() for i in range(self.file_list.count())]
        analysis_type = self.analysis_type.currentText()
        files = [self.file_list.item(i).text() for i in range(self.file_list.count())]
        self.result_display.clear()
        self.noise_levels = []
        self.snr_levels = []
        self.clipping_data = []
        all_files_valid = True

        invalid_results = ""  # Sadece INVALID dosyalar için birikmeli sonuçlar

        for i, file_path in enumerate(selected_files):
            QApplication.processEvents()
            file_name = os.path.basename(file_path)
            result = f"<b>Analyzed File Name: {file_name}</b><br>"
            valid_file = True
            invalid_reasons = []

            if analysis_type == "Verify Format and Sampling Rate":
                format_ok, file_format = self.audio_checker.check_format(file_path)
                rate_ok, file_rate = self.audio_checker.check_sampling_rate(file_path, self.target_rates)
                result += f"Format: {file_format} (Supported: {format_ok})<br>"
                result += f"Sampling Rate: {file_rate}Hz (Accepted: {rate_ok})<br>"
                if not format_ok:
                    invalid_reasons.append("Unsupported Format")
                if not rate_ok:
                    invalid_reasons.append("Invalid Sampling Rate")
                valid_file &= format_ok and rate_ok

            elif analysis_type == "Analyze Background Noise":
                noise_level, acceptable = self.audio_checker.calculate_rms(file_path)
                self.noise_levels.append(noise_level)
                result += f"RMS Noise Level: {noise_level}dB (Acceptable: {acceptable})<br>"
                if not acceptable:
                    invalid_reasons.append("High Background Noise")
                valid_file &= acceptable

            elif analysis_type == "Analyze SNR":
                snr, acceptable = self.audio_checker.calculate_snr(file_path)
                self.snr_levels.append(snr)
                result += f"SNR: {snr}dB (Acceptable: {acceptable})<br>"
                if not acceptable:
                    invalid_reasons.append("Low SNR")
                valid_file &= acceptable

            elif analysis_type == "Detect Clipping":
                clipping, points = self.audio_checker.detect_clipping(file_path)
                result += f"Clipping Detected: {clipping} (Points: {len(points)})<br>"
                if clipping:
                    self.clipping_data.append((file_name, points))
                    invalid_reasons.append("Clipping Detected")
                valid_file &= not clipping

            elif analysis_type == "Analyze Reverb":
                rt60 = self.audio_checker.calculate_reverb(file_path)
                result += f"Reverb Time (RT60): {rt60}<br>"
                if rt60 >= 2:
                    invalid_reasons.append("High Reverb Time (RT60)")
                valid_file &= rt60 < 2

            elif analysis_type == "Inspect Channel Mode":
                channel_mode, num_channels = self.audio_checker.check_channel_mode(file_path)
                result += f"Channel Mode: {channel_mode} (Channels: {num_channels})<br>"
                if channel_mode not in ["stereo", "mono"]:
                    invalid_reasons.append("Invalid Channel Mode")
                self.channel_modes.append(channel_mode)
                valid_file &= (channel_mode == "stereo" or channel_mode == "mono")

            elif analysis_type == "Verify Bit Depth":
                bit_depth, valid = self.audio_checker.check_bit_depth(file_path)
                result += f"Bit Depth: {bit_depth} (Valid: {valid})<br>"
                if not valid:
                    invalid_reasons.append("Invalid Bit Depth")
                valid_file &= valid

            # Sadece INVALID dosyaları ekle ve INVALID sebeplerini vurgula
            if not valid_file:
                all_files_valid = False
                result += f"<b>Status: <span style='color: red;'>INVALID FILE</span></b><br>"
                if invalid_reasons:
                    reasons = "<br>".join(
                        [f"<b><span style='background-color: yellow;'>{reason}</span></b>" for reason in
                         invalid_reasons])
                    result += f"Reasons:<br>{reasons}<br><br>"
                invalid_results += result + "<br>"

            self.current_analysis_type = analysis_type
            # Progress Bar'ı güncelle
            progress_value = int(((i + 1) / len(files)) * 100)
            self.progress_bar.setValue(progress_value)

        # Sonuçları ekrana göster
        if invalid_results:
            self.result_display.setHtml(invalid_results)
        else:
            self.result_display.setHtml("<b>All files are valid.</b>")

        if all_files_valid:
            self.result_display.setStyleSheet("background-color: lightgreen;")
        else:
            self.result_display.setStyleSheet("background-color: lightcoral;")
        QApplication.processEvents()

    def perform_all_analyses(self):
        QApplication.processEvents()
        files = [self.file_list.item(i).text() for i in range(self.file_list.count())]
        self.result_display.clear()
        invalid_results = ""
        all_files_valid = True

        for i, file_path in enumerate(files):
            QApplication.processEvents()
            file_name = os.path.basename(file_path)
            result = f"<b>Analyzed File Name: {file_name}</b><br>"
            valid_file = True
            invalid_reasons = []

            format_ok, file_format = self.audio_checker.check_format(file_path)
            rate_ok, file_rate = self.audio_checker.check_sampling_rate(file_path, self.target_rates)
            result += f"Format: {file_format} (Supported: {format_ok})<br>"
            result += f"Sampling Rate: {file_rate}Hz (Accepted: {rate_ok})<br>"
            if not format_ok:
                invalid_reasons.append("Unsupported Format")
            if not rate_ok:
                invalid_reasons.append("Invalid Sampling Rate")
            valid_file &= format_ok and rate_ok

            noise_level, acceptable = self.audio_checker.calculate_rms(file_path)
            result += f"RMS Noise Level: {noise_level}dB (Acceptable: {acceptable})<br>"
            self.noise_levels.append(noise_level)
            if not acceptable:
                invalid_reasons.append("High Background Noise")
            valid_file &= acceptable

            snr, acceptable = self.audio_checker.calculate_snr(file_path)
            result += f"SNR: {snr}dB (Acceptable: {acceptable})<br>"
            self.snr_levels.append(snr)
            if not acceptable:
                invalid_reasons.append("Low SNR")
            valid_file &= acceptable

            clipping, points = self.audio_checker.detect_clipping(file_path)
            result += f"Clipping Detected: {clipping} (Points: {len(points)})<br>"
            if clipping:
                invalid_reasons.append("Clipping Detected")
                self.clipping_data.append((file_name, points))
            valid_file &= not clipping

            rt60 = self.audio_checker.calculate_reverb(file_path)
            result += f"Reverb Time (RT60): {rt60}<br>"
            if rt60 >= 2:
                invalid_reasons.append("High Reverb Time (RT60)")
            valid_file &= rt60 < 2

            channel_mode, num_channels = self.audio_checker.check_channel_mode(file_path)
            result += f"Channel Mode: {channel_mode} (Channels: {num_channels})<br>"
            self.channel_modes.append(channel_mode)

            if channel_mode not in ["stereo", "mono"]:
                invalid_reasons.append("Invalid Channel Mode")
            valid_file &= (channel_mode == "stereo" or channel_mode == "mono")

            bit_depth, valid = self.audio_checker.check_bit_depth(file_path)
            result += f"Bit Depth: {bit_depth} (Valid: {valid})<br>"
            if not valid:
                invalid_reasons.append("Invalid Bit Depth")
            valid_file &= valid
            self.current_analysis_type = "All"

            if not valid_file:
                all_files_valid = False
                result += f"<b>Status: <span style='color: red;'>INVALID FILE</span></b><br>"
                if invalid_reasons:
                    reasons = "<br>".join(
                        [f"<b><span style='background-color: yellow;'>{reason}</span></b>" for reason in
                         invalid_reasons])
                    result += f"Reasons:<br>{reasons}<br><br>"
                invalid_results += result + "<br>"

            progress_value = int(((i + 1) / len(files)) * 100)

            self.progress_bar.setValue(progress_value)
            if all_files_valid:
                self.result_display.setStyleSheet("background-color: lightgreen;")
            else:
                self.result_display.setStyleSheet("background-color: lightcoral;")

            QApplication.processEvents()

            if invalid_results:
                self.result_display.setHtml(invalid_results)
            else:
                self.result_display.setHtml("<b>All files are valid.</b>")

                self.result_text = self.result_display.toPlainText()
                progress_value = int(((i + 1) / len(files)) * 100)
                self.progress_bar.setValue(progress_value)

    def download_pdf(self):
        try:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(self, "Save PDF", "", "PDF Files (*.pdf);;All Files (*)",
                                                       options=options)
            if not file_path:
                return
            if not file_path.endswith('.pdf'):
                file_path += '.pdf'

            pdf_canvas = canvas.Canvas(file_path, pagesize=letter)
            text = self.result_text

            if not text.strip():
                self.result_display.append("\nNo analysis results to download.")
                return

            pdf_canvas.drawString(100, 750, "Audio File Inspection Results")
            lines = text.split('\n')
            y_position = 700

            for line in lines:
                if y_position < 50:
                    pdf_canvas.showPage()
                    y_position = 750
                pdf_canvas.drawString(100, y_position, line)
                y_position -= 20

            pdf_canvas.save()

            self.result_display.append("\nPDF successfully saved at: " + file_path)

        except Exception as e:
            self.result_display.append("\nError while saving PDF: " + str(e))

    def download_excel(self):
        try:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Excel", "", "Excel Files (*.xlsx);;All Files (*)",
                                                       options=options)

            if not file_path:
                return

            if not file_path.endswith('.xlsx'):
                file_path += '.xlsx'

            text = self.result_text

            if not text.strip():
                self.result_display.append("\nNo analysis results to download.")
                return

            lines = text.split('\n')
            data = [line.split() for line in lines if line.strip()]

            df = pd.DataFrame(data)
            df.to_excel(file_path, index=False, header=False)

            self.result_display.append("\nExcel successfully saved at: " + file_path)

        except Exception as e:
            self.result_display.append("\nError while saving Excel: " + str(e))

    def download_csv(self):
        try:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv);;All Files (*)",
                                                       options=options)

            if not file_path:
                return

            if not file_path.endswith('.csv'):
                file_path += '.csv'

            text = self.result_text

            if not text.strip():
                self.result_display.append("\nNo analysis results to download.")
                return

            lines = text.split('\n')
            data = [line.split() for line in lines if line.strip()]
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False, header=False)

            self.result_display.append("\nCSV successfully saved at: " + file_path)

        except Exception as e:
            self.result_display.append("\nError while saving CSV: " + str(e))

    def download_statistics(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(self, "Save Statistics", "", "PNG Files (*.png)", options=options)

        if filename and (len(self.noise_levels) > 0 or len(self.snr_levels) > 0 or len(self.clipping_data) > 0 or len(
                self.channel_modes) > 0):

            plt.figure(figsize=(15, 10))
            if self.current_analysis_type == "All":
                # Noise Levels Plot
                plt.subplot(2, 2, 1)
                plt.bar(range(len(self.noise_levels)), self.noise_levels, color='blue')
                plt.title('Noise Levels')
                plt.xlabel('File Number')
                plt.ylabel('Level (dB)')

                # SNR Levels Plot
                plt.subplot(2, 2, 2)
                plt.bar(range(len(self.snr_levels)), self.snr_levels, color='orange')
                plt.title('SNR Levels')
                plt.xlabel('File Number')
                plt.ylabel('Level (dB)')

                # Clipping Plot
                plt.subplot(2, 2, 3)
                plt.plot(self.clipping_data, label='Clipping', color='red')
                plt.title('Clipping')
                plt.xlabel('File Number')
                plt.ylabel('Clipping Level')

                # Channel Modes Plot
                plt.subplot(2, 2, 4)
                labels = ['Mono' if mode == "mono" else "Stereo" for mode in self.channel_modes]
                unique_modes, counts = np.unique(labels, return_counts=True)
                plt.bar(unique_modes, counts, color='orange')
                plt.title('Channel Modes')
                plt.ylabel('Counts')
                plt.xlabel('Channel Type')

                plt.tight_layout()

            elif self.current_analysis_type == "Analyze SNR":
                plt.bar(range(len(self.snr_levels)), self.snr_levels, color='green')
                plt.title('SNR Levels')
                plt.ylabel('Level (dB)')
                plt.xlabel('File Number')

            elif self.current_analysis_type == "Detect Clipping":
                plt.plot(self.clipping_data, label='Clipping', color='red')
                plt.title('Clipping Detection')
                plt.ylabel('Clipping Level')
                plt.xlabel('File Number')

            elif self.current_analysis_type == "Inspect Channel Mode":
                labels = ['Mono' if mode == "mono" else "Stereo" for mode in self.channel_modes]
                unique_modes, counts = np.unique(labels, return_counts=True)
                plt.bar(unique_modes, counts, color='orange')
                plt.title('Channel Modes')
                plt.ylabel('Counts')
                plt.xlabel('Channel Type')

            elif self.current_analysis_type == "Analyze Background Noise":
                plt.bar(range(len(self.noise_levels)), self.noise_levels, color='blue')
                plt.title('Noise Levels')
                plt.ylabel('Level (dB)')
                plt.xlabel('File Number')

            plt.savefig(filename)
            plt.close()

    def show_statistics(self):
        previous_results = self.result_display.toHtml()

        self.result_display.clear()

        plt.figure(figsize=(15, 10))

        if self.current_analysis_type == "All":
            # Noise Levels Plot
            plt.subplot(2, 2, 1)
            plt.bar(range(len(self.noise_levels)), self.noise_levels, color='blue')
            plt.title('Noise Levels')
            plt.xlabel('File Index')
            plt.ylabel('Level (dB)')
            plt.xticks(range(len(self.noise_levels)), rotation=45)

            # SNR Levels Plot
            plt.subplot(2, 2, 2)
            plt.bar(range(len(self.snr_levels)), self.snr_levels, color='orange')
            plt.title('SNR Levels')
            plt.xlabel('File Index')
            plt.ylabel('Level (dB)')
            plt.xticks(range(len(self.snr_levels)), rotation=45)

            # Clipping Plot
            plt.subplot(2, 2, 3)
            plt.plot(self.clipping_data, label='Clipping', color='red')
            plt.title('Clipping')
            plt.xlabel('File Index')
            plt.ylabel('Clipping Level')
            plt.xticks(range(len(self.clipping_data)), rotation=45)

            # Channel Modes Plot
            plt.subplot(2, 2, 4)
            labels = ['Mono' if mode == "mono" else "Stereo" for mode in self.channel_modes]
            unique_modes, counts = np.unique(labels, return_counts=True)
            plt.bar(unique_modes, counts, color='orange')
            plt.title('Channel Modes')
            plt.ylabel('Counts')
            plt.xlabel('Channel Type')

        elif self.current_analysis_type == "Analyze SNR":
            plt.bar(range(len(self.snr_levels)), self.snr_levels, color='green')
            plt.title('SNR Levels')
            plt.ylabel('Level (dB)')
            plt.xlabel('File Index')
            plt.xticks(range(len(self.snr_levels)), rotation=45)

        elif self.current_analysis_type == "Detect Clipping":
            plt.plot(self.clipping_data, label='Clipping', color='red')
            plt.title('Clipping Detection')
            plt.ylabel('Clipping Level')
            plt.xlabel('File Index')
            plt.xticks(range(len(self.clipping_data)), rotation=45)

        elif self.current_analysis_type == "Inspect Channel Mode":
            labels = ['Mono' if mode == "mono" else "Stereo" for mode in self.channel_modes]
            unique_modes, counts = np.unique(labels, return_counts=True)
            plt.bar(unique_modes, counts, color='orange')
            plt.title('Channel Modes')
            plt.ylabel('Counts')
            plt.xlabel('Channel Type')

        elif self.current_analysis_type == "Analyze Background Noise":
            plt.bar(range(len(self.noise_levels)), self.noise_levels, color='blue')
            plt.title('Noise Levels')
            plt.ylabel('Level (dB)')
            plt.xlabel('File Index')
            plt.xticks(range(len(self.noise_levels)), rotation=45)

        plt.tight_layout()

        # Save the plot to a file
        plt.savefig('plot.png', bbox_inches='tight', dpi=150)
        plt.close()

        # Display the plot in result_display
        pixmap = QPixmap('plot.png')
        self.result_display.insertHtml('<img src="plot.png" width="800" />')

        # Metin sonuçlarını tekrar ekleyelim
        self.result_display.insertHtml(previous_results)

        # Delete the saved plot file after displaying
        if os.path.exists('plot.png'):
            os.remove('plot.png')

    def remove_selected_files(self):
        for item in self.file_list.selectedItems():
            self.file_list.takeItem(self.file_list.row(item))

    def resource_path(self, relative_path):
        try:
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AudioInspectorApp()
    window.show()
    sys.exit(app.exec_())
