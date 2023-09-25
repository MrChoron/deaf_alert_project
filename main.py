
import sys
import librosa
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QTextBrowser, QMessageBox
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

class AudioClassificationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Classification")

        # Load your trained model
        self.model = load_model(r'C:savedmodelpath.hdf5')

        # Create a LabelEncoder instance
        self.labelencoder = LabelEncoder()

        # Fit the LabelEncoder with class labels
        class_names = [
            "air_conditioner", "car_horn", "children_playing",
            "dog_bark", "drilling", "engine_idling",
            "gun_shot", "jackhammer", "siren", "street_music"
        ]
        self.labelencoder.fit(class_names)  # Fit with class labels

        # Create GUI elements
        self.select_button = QPushButton("Select Audio File", self)
        self.select_button.setGeometry(100, 20, 200, 30)
        self.select_button.clicked.connect(self.load_audio_file)

        self.result_label = QLabel(self)
        self.result_label.setGeometry(100, 70, 200, 30)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("background-color: lightgray;")

        self.result_display = QTextBrowser(self)
        self.result_display.setGeometry(50, 120, 300, 120)
        self.result_display.setPlainText("Predicted Class:")

        self.audio_path = ""

    def load_audio_file(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(None, "Select Audio File", "", "Audio Files (*.wav)")

        if file_path:
            self.audio_path = file_path
            self.predict_audio_class()

    def predict_audio_class(self):
        if self.audio_path:
            # Load the audio file and extract MFCC features
            audio, sample_rate = librosa.load(self.audio_path, res_type='kaiser_fast')
            mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

            # Calculate the mean of MFCCs
            mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)

            # Reshape the features to match the input shape expected by the model
            mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)

            # Make predictions using the model
            predicted_probabilities = self.model.predict(mfccs_scaled_features)

            # Find the class with the highest probability
            predicted_label = np.argmax(predicted_probabilities, axis=1)

            # Inverse transform the predicted label to get the class name
            prediction_class = self.labelencoder.inverse_transform(predicted_label)

            self.result_label.setText(f"Predicted Class: {prediction_class[0]}")
            self.result_display.setPlainText(f"Predicted Class: {prediction_class[0]}")

            # Check if the predicted class is "dog_bark," "gun_shot," or "siren"
            if prediction_class[0] in ["dog_bark", "gun_shot", "siren"]:
                # Change the background color of the result_label to alert the user
                self.result_label.setStyleSheet("background-color: red;")
                # You can also display a message box with an alert
                alert_message = f"Alert: Predicted class is {prediction_class[0]}!"
                QMessageBox.warning(self, "Alert", alert_message)
            else:
                # Reset the background color if it's not one of the alert classes
                self.result_label.setStyleSheet("background-color: lightgray;")
        else:
            self.result_label.setText("No audio file selected.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioClassificationApp()
    window.setGeometry(100, 100, 400, 300)
    window.show()
    sys.exit(app.exec())


