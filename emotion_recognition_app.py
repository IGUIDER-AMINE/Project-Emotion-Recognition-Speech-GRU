import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import os

# Définir les étiquettes d'émotion
emotion_labels = {
    0: "Neutral",
    1: "Happy",
    2: "Sad",
    3: "Angry",
    4: "Fear",
    5: "Disgust",
    6: "Surprise"
}

# Fonction pour prétraiter l'audio
def preprocess_audio(path, target_sr=22050, n_mels=15, expected_time_steps=352):
    """
    Prétraiter un fichier audio pour la reconnaissance des émotions en extrayant des caractéristiques Mel-spectrogramme.
    """
    # Charger l'audio
    audio, sr = librosa.load(path, sr=target_sr)

    # Retirer les silences
    trimmed, _ = librosa.effects.trim(audio, top_db=25)

    # Extraire le Mel-spectrogramme
    mel_spectrogram = librosa.feature.melspectrogram(y=trimmed, sr=target_sr, n_mels=n_mels)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Transposer et ajuster la taille
    mel_spectrogram_db = mel_spectrogram_db.T
    if mel_spectrogram_db.shape[0] < expected_time_steps:
        padding = expected_time_steps - mel_spectrogram_db.shape[0]
        mel_spectrogram_db = np.pad(mel_spectrogram_db, ((0, padding), (0, 0)), mode='constant')
    else:
        mel_spectrogram_db = mel_spectrogram_db[:expected_time_steps, :]

    # Ajouter une dimension de batch
    return np.expand_dims(mel_spectrogram_db, axis=0)

# Fonction pour prédire l'émotion
def predict_emotion(audio_file_path):
    """
    Prédire l'émotion d'un fichier audio à l'aide d'un modèle GRU préentraîné.
    """
    try:
        # Charger le modèle
        model = load_model("model_gru.h5")

        # Prétraiter le fichier audio
        input_data = preprocess_audio(audio_file_path, expected_time_steps=352)

        # Faire une prédiction
        predictions = model.predict(input_data)
        predicted_emotion_index = np.argmax(predictions)
        confidence = np.max(predictions)

        # Mapper l'indice à une étiquette
        predicted_emotion = emotion_labels[predicted_emotion_index]
        return predicted_emotion, confidence
    except Exception as e:
        raise RuntimeError(f"Erreur lors de la prédiction : {e}")

# Application Streamlit
def main():
    st.title("Reconnaissance des émotions dans les fichiers audio")
    st.write("Téléchargez un fichier audio pour analyser l'émotion.")

    # Téléchargement du fichier
    uploaded_file = st.file_uploader("Choisissez un fichier audio (.wav uniquement)", type=["wav"])

    if uploaded_file is not None:
        # Sauvegarder temporairement le fichier
        temp_audio_file = "temp_audio_file.wav"
        with open(temp_audio_file, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Afficher le fichier audio
        st.audio(temp_audio_file, format="audio/wav")

        # Prétraiter et prédire l'émotion
        try:
            st.write("**Analyse en cours...**")
            predicted_emotion, confidence = predict_emotion(temp_audio_file)
            st.success(f"**Émotion prédite :** {predicted_emotion}")
            st.write(f"**Score de confiance :** {confidence:.2f}")
        except Exception as e:
            st.error(f"Une erreur est survenue : {e}")

        # Supprimer le fichier temporaire
        if os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)

if __name__ == "__main__":
    main()
