# Emotion-Based Song Recommendation System 🎵🙂

An AI-powered web app that detects your facial emotion using your webcam and recommends music that matches your mood. Built with TensorFlow, OpenCV, and Streamlit. Integrated with Spotify and YouTube for real-time previews.

## 💡 Features

- Real-time facial emotion detection using webcam.
- Emotion-based song suggestions (happy, sad, angry, etc.).
- Integration with Spotify and YouTube for previews.
- Simple, user-friendly Streamlit interface.
- Two modes: Quick Scan & Live Emotion Tracking.

## 🛠️ Tech Stack

- **Python**
- **TensorFlow/Keras**
- **OpenCV**
- **Streamlit**
- **Pandas**
- **Spotify API**
- **YouTube Search API**

## 📁 Files Included

- `app.py` - Main Streamlit app
- `model.h5` - Trained CNN model
- `muse_v3.csv` - Song dataset
- `spotify_api.py` - Spotify integration script
- `youtube_api.py` - YouTube integration script
- `utils.py` - Helper functions (face detection, emotion classification)

## 🧠 Emotion Detection

The CNN model is trained to classify 7 basic emotions:  
`Happy`, `Sad`, `Angry`, `Fear`, `Surprise`, `Neutral`, `Disgust`

## How to install 

```bash
git clone https://github.com/yourusername/emotion-music-recommender.git
cd emotion-music-recommender


python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py

