import streamlit as st
import numpy as np
import librosa
import os
from tensorflow.keras.models import load_model

# Définir les labels des émotions
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
    Prétraiter un fichier audio en extrayant des caractéristiques de Mel-spectrogramme.

    Parameters:
        path (str): Chemin du fichier audio.
        target_sr (int): Fréquence d'échantillonnage cible.
        n_mels (int): Nombre de caractéristiques Mel.
        expected_time_steps (int): Nombre attendu de pas de temps.

    Returns:
        np.ndarray: Mel-spectrogramme prétraité.
    """
    try:
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

        # Ajouter une dimension pour le batch
        return np.expand_dims(mel_spectrogram_db, axis=0)
    except Exception as e:
        raise RuntimeError(f"Erreur lors du prétraitement de l'audio : {e}")

# Fonction pour prédire l'émotion
def predict_emotion(audio_file_path, model):
    """
    Prédire l'émotion d'un fichier audio.

    Parameters:
        audio_file_path (str): Chemin du fichier audio.
        model: Modèle TensorFlow entraîné.

    Returns:
        tuple: Émotion prédite et score de confiance.
    """
    try:
        # Prétraiter l'audio
        input_data = preprocess_audio(audio_file_path, expected_time_steps=352)

        # Faire une prédiction
        predictions = model.predict(input_data)
        predicted_emotion_index = np.argmax(predictions)
        confidence = np.max(predictions)

        # Mapper l'indice à une étiquette d'émotion
        predicted_emotion = emotion_labels[predicted_emotion_index]
        return predicted_emotion, confidence
    except Exception as e:
        raise RuntimeError(f"Erreur lors de la prédiction : {e}")

# Application Streamlit
def main():
    st.title("Reconnaissance des émotions dans les signaux vocaux")
    st.write("Téléchargez un fichier audio pour prédire l'émotion associée.")

    # Chargement de fichier
    uploaded_file = st.file_uploader("Choisissez un fichier audio (.wav ou .mp3)", type=["wav", "mp3"])

    if uploaded_file is not None:
        # Enregistrer le fichier temporairement
        temp_audio_file = "temp_audio_file.wav"
        with open(temp_audio_file, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Afficher le fichier audio dans Streamlit
        st.audio(temp_audio_file, format="audio/wav")
        st.write("**Analyse en cours...**")

        try:
            # Vérification du fichier audio
            audio, sr = librosa.load(temp_audio_file, sr=None)
            duration = len(audio) / sr
            if duration < 1.0:
                st.error("Le fichier audio est trop court pour une reconnaissance fiable des émotions.")
                return

            # Charger le modèle
            model = load_model("model_gru.h5")

            # Prédire l'émotion
            with st.spinner("Prédiction en cours..."):
                predicted_emotion, confidence = predict_emotion(temp_audio_file, model)

            # Afficher les résultats
            st.success(f"**Émotion prédite :** {predicted_emotion}")
            st.write(f"**Score de confiance :** {confidence:.2f}")

        except Exception as e:
            st.error(f"Une erreur est survenue : {e}")

        # Supprimer le fichier temporaire
        if os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)

if __name__ == "__main__":
    main()
