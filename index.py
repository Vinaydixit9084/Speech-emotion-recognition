import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import soundfile as sf
from scipy.signal import resample
from python_speech_features import mfcc, logfbank

# Load the dataset
dataset_path = r'C:\Users\DELL\Desktop\CodersCave project\Speech emotion detection\audio&csv\speech_emotions.csv'

df = pd.read_csv(dataset_path)

# Function to extract features from audio files
def extract_features(file_path):
    try:
        # Load audio file with soundfile
        audio, sr = sf.read(file_path)

        # Downsample the audio to a common sample rate (if needed)
        target_sr = 16000
        audio = resample(audio, int(audio.shape[0] * target_sr / sr))

        # Extract features using python_speech_features
        mfccs = np.mean(mfcc(audio, samplerate=target_sr, nfft=1200).T, axis=0)
        logfbank_features = np.mean(logfbank(audio, samplerate=target_sr, nfft=1200).T, axis=0)

        return np.hstack((mfccs, logfbank_features))
    except Exception as e:
        print(f"file processed {file_path}: {e}")
        return None


# Apply feature extraction to each row in the DataFrame (use 'text' column instead of 'file_path')
df['features'] = df['text'].apply(extract_features)

# Remove rows with missing features
df = df.dropna()

# Check if there are still rows in the DataFrame
if df.empty:
    print("No valid samples remaining after feature extraction.")
else:
    # Split the dataset into training and testing sets
    X = np.vstack(df['features'].to_numpy())
    y = df['emotion']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a simple Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy}')
    print(f'Confusion Matrix:\n{conf_matrix}')
    print(f'Classification Report:\n{classification_rep}')
