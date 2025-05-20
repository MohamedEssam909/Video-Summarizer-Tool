import sys
import cv2
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QProgressBar, QSpinBox, QHBoxLayout, QScrollArea, QGridLayout
)
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal


from video_summarizer import (
    extract_frames, detect_scene_changes, split_video_by_scenes, create_summary_video
)


import moviepy.config as mpy_config


if getattr(sys, 'frozen', False):
    ffmpeg_path = os.path.join(sys._MEIPASS, 'ffmpeg.exe')  
else:
    ffmpeg_path = 'ffmpeg.exe'  

os.environ["FFMPEG_BINARY"] = ffmpeg_path
mpy_config.FFMPEG_BINARY = ffmpeg_path

class Worker(QThread):
    progress = pyqtSignal(int)
    done = pyqtSignal()
    load_thumbnails_signal = pyqtSignal(str)  # New signal to load thumbnails

    def __init__(self, video_path, top_k, ui_ref, output_path):  
        super().__init__()
        self.video_path = video_path
        self.top_k = top_k
        self.ui = ui_ref
        self.output_path = output_path  

    def run(self):
        frames_folder = "frames"
        scenes_folder = "scenes"
        summary_output = "summary.mp4"

        extract_frames(self.video_path, frames_folder, fps=1)
        self.progress.emit(25)

        # Wait until frames are fully extracted before proceeding with scene detection
        scene_changes = detect_scene_changes(frames_folder)
        print(f"Scene Changes Detected: {scene_changes}")

        if not scene_changes:  
            print("No scene changes detected.")
            self.ui.label.setText("No scene changes detected.")
            self.done.emit()
            return  

        self.progress.emit(50)

        split_video_by_scenes(self.video_path, frames_folder, scene_changes, scenes_folder)
        
        # Emit signal to load thumbnails from frames folder
        self.load_thumbnails_signal.emit(frames_folder)
        
        self.progress.emit(75)

        create_summary_video(scenes_folder, self.output_path, top_k=self.top_k)
        self.progress.emit(100)

        self.done.emit()



class VideoSummarizerGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Summarizer Tool")
        self.setGeometry(300, 100, 800, 600)
        self.setWindowIcon(QIcon("favicon.ico"))  

        self.setStyleSheet("""
            QWidget {
                background-color: #121212;
                font-family: "Segoe UI", sans-serif;
                font-size: 14px;
                color: #f0f0f0;
            }

            QLabel {
                font-size: 15px;
                padding: 6px 0;
                color: #e0e0e0;
            }

            QPushButton {
                background-color: #1f1f1f;
                color: #ffffff;
                padding: 10px 20px;
                border: 1px solid #333;
                border-radius: 10px;
                font-weight: 600;
            }

            QPushButton:hover {
                background-color: #2a2a2a;
                border: 1px solid #555;
            }

            QPushButton:pressed {
                background-color: #3a3a3a;
            }

            QProgressBar {
                height: 20px;
                border-radius: 10px;
                background-color: #2b2b2b;
                border: 1px solid #444;
            }

            QProgressBar::chunk {
                border-radius: 10px;
                background-color: #4caf50;
            }

            QSpinBox {
                background-color: #1c1c1c;
                color: #f0f0f0;
                border: 1px solid #444;
                border-radius: 6px;
                padding: 4px 10px;
                min-width: 70px;
            }

            QSpinBox::up-button, QSpinBox::down-button {
                background-color: #2a2a2a;
                border-radius: 4px;
                width: 16px;
            }

            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background-color: #3a3a3a;
            }

            QScrollArea {
                background-color: #1a1a1a;
                border: 1px solid #333;
                border-radius: 8px;
            }
        """)




        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.label = QLabel("Choose a video file to summarize:")
        self.layout.addWidget(self.label)

        self.pick_btn = QPushButton("Select Video")
        self.pick_btn.setMaximumWidth(200)
        self.pick_btn.clicked.connect(self.pick_video)
        self.layout.addWidget(self.pick_btn)

        self.topk_layout = QHBoxLayout()
        self.layout.addLayout(self.topk_layout)

        self.topk_label = QLabel("Summary length (Top K scenes):")
        self.topk_layout.addWidget(self.topk_label)

        self.topk_input = QSpinBox()
        self.topk_input.setMinimum(1)
        self.topk_input.setValue(5)
        self.topk_layout.addWidget(self.topk_input)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.layout.addWidget(self.progress)

        self.thumb_area = QScrollArea()
        self.thumb_widget = QWidget()
        self.thumb_layout = QGridLayout()
        self.thumb_widget.setLayout(self.thumb_layout)
        self.thumb_area.setWidgetResizable(True)
        self.thumb_area.setWidget(self.thumb_widget)
        self.layout.addWidget(QLabel("Frames Preview:"))
        self.layout.addWidget(self.thumb_area)

    def pick_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Videos (*.mp4 *.avi *.mov)")
        
        if path:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                self.label.setText("Error: Unable to open video file.")
                return  

            self.label.setText(f"Processing: {os.path.basename(path)}")
            self.progress.setValue(0)
            top_k = self.topk_input.value()

            output_path, _ = QFileDialog.getSaveFileName(self, "Choose where to save summary video", "", "Videos (*.mp4)")
            if not output_path:
                self.label.setText("Error: No output path selected.")
                return  

            self.worker = Worker(path, top_k, self, output_path)
            self.worker.progress.connect(self.progress.setValue)
            self.worker.done.connect(lambda: self.label.setText(f"Summary saved to {output_path}"))
            
            # Connect the new signal to load thumbnails
            self.worker.load_thumbnails_signal.connect(self.load_thumbnails)
            
            self.worker.start()

    def load_thumbnails(self, frames_folder):
        # Clear the previous thumbnails
        for i in reversed(range(self.thumb_layout.count())):
            self.thumb_layout.itemAt(i).widget().setParent(None)

        # Get the list of frame files (sorted by filename)
        frame_files = sorted(f for f in os.listdir(frames_folder) if f.endswith(('.jpg', '.png')))

        for idx, filename in enumerate(frame_files):
            # Construct the full path to the frame image
            thumb_path = os.path.join(frames_folder, filename)

            if os.path.exists(thumb_path):  # Check if the frame file exists
                pixmap = QPixmap(thumb_path).scaledToWidth(160, Qt.SmoothTransformation)
                thumb_label = QLabel()
                thumb_label.setPixmap(pixmap)
                thumb_label.setStyleSheet("""
                    background-color: #1e1e1e;
                    border: 1px solid #333;
                    border-radius: 8px;
                    padding: 6px;
                    margin: 6px;
                """)
                self.thumb_layout.addWidget(thumb_label, idx // 4, idx % 4)
            else:
                print(f"Frame not found: {thumb_path}")

                
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoSummarizerGUI()
    window.show()
    sys.exit(app.exec_())
