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

## Getting API Credentials

### 1. Spotify Developer Credentials

1. Go to the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/applications).
2. Log in with your Spotify account or create one.
3. Click **"Create an App"**.
4. Fill in the app name and description.
5. After creating, you'll see **Client ID** and **Client Secret** on your app page.
6. Copy those and add them to your project’s `.env` or config file as:

#POTIFY_CLIENT_ID=your_client_id_here
#SPOTIFY_CLIENT_SECRET=your_client_secret_here

### 2. YouTube API Key Setup

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).

2. Create a new project or select an existing one.

3. In the left sidebar, navigate to **APIs & Services > Library**.

4. Search for **YouTube Data API v3** and click **Enable**.

5. Then go to **APIs & Services > Credentials**.

6. Click **Create Credentials > API Key**.

7. Copy the generated API Key.

8. Add the API Key to your `.env` or config file like this:


#YOUTUBE_API_KEY=your_api_key_here


## How to install 

```bash
git clone https://github.com/yourusername/emotion-music-recommender.git
cd emotion-music-recommender
``` 
## make a Environment for python 3.11 

```bash
python -m venv venv
source venv/bin/activate
```

```bash
pip install -r requirements.txt
streamlit run final.py
```
## 📸 Screenshots

### Home Page
![Home](assets/screenshot(565).png)

### Emotion Detection (Quick Scan)
![Emotion Detection](assets/screenshot(563).png)

### Recommended Songs with Youtube previw 
![Songs](assets/screenshot(534).png)

