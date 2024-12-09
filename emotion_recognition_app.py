import streamlit as st
import numpy as np
import librosa
from pydub import AudioSegment
from tensorflow.keras.models import load_model

# Define emotion labels
emotion_labels = {
    0: "Neutral",
    1: "Happy",
    2: "Sad",
    3: "Angry",
    4: "Fear",
    5: "Disgust",
    6: "Surprise"
}

# Function to preprocess audio
def preprocess_audio(path, target_sr=22050, n_mels=15, expected_time_steps=352):
    """
    Preprocess audio for emotion recognition by extracting Mel-spectrogram features.

    Parameters:
        path (str): Path to the audio file.
        target_sr (int): Target sampling rate.
        n_mels (int): Number of Mel-frequency features.
        expected_time_steps (int): Expected number of time steps.

    Returns:
        np.ndarray: Processed Mel-spectrogram features.
    """
    # Load audio
    audio, sr = librosa.load(path, sr=target_sr)

    # Trim silence
    trimmed, _ = librosa.effects.trim(audio, top_db=25)

    # Extract Mel-spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=trimmed, sr=target_sr, n_mels=n_mels)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Transpose to match time steps and truncate or pad to expected time steps
    mel_spectrogram_db = mel_spectrogram_db.T  # Transpose to shape (time_steps, n_mels)
    if mel_spectrogram_db.shape[0] < expected_time_steps:
        padding = expected_time_steps - mel_spectrogram_db.shape[0]
        mel_spectrogram_db = np.pad(mel_spectrogram_db, ((0, padding), (0, 0)), mode='constant')
    else:
        mel_spectrogram_db = mel_spectrogram_db[:expected_time_steps, :]

    # Add batch dimension
    return np.expand_dims(mel_spectrogram_db, axis=0)

# Function to predict emotion
def predict_emotion(audio_file_path, model):
    """
    Predict the emotion of an audio file using the trained GRU model.

    Parameters:
        audio_file_path (str): Path to the audio file.
        model: Trained TensorFlow model.

    Returns:
        tuple: Predicted emotion label and confidence score.
    """
    # Preprocess the audio
    input_data = preprocess_audio(audio_file_path, expected_time_steps=352)

    # Predict emotion
    predictions = model.predict(input_data)
    predicted_emotion_index = np.argmax(predictions)
    confidence = np.max(predictions)

    # Get the emotion label
    predicted_emotion = emotion_labels[predicted_emotion_index]
    return predicted_emotion, confidence

# Streamlit app
def main():
    st.title("Emotion Recognition from Speech Signals")
    st.write("Upload an audio file to predict the emotion.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

    if uploaded_file is not None:
        # Save the uploaded file
        with open("temp_audio_file.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.audio("temp_audio_file.wav", format="audio/wav")
        st.write("Processing the uploaded file...")

        try:
            # Load the model
            model = load_model("model_gru.h5")

            # Predict emotion
            predicted_emotion, confidence = predict_emotion("temp_audio_file.wav", model)

            # Display results
            st.write(f"**Predicted Emotion:** {predicted_emotion}")
            st.write(f"**Confidence Score:** {confidence:.2f}")

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()