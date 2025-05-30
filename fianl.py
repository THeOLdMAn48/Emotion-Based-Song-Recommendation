import numpy as np
import streamlit as st
import cv2
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests

# --- CREDENTIALS ---
SPOTIFY_CLIENT_ID = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" #read README.md to make
SPOTIFY_CLIENT_SECRET = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" #read README.md to make
YOUTUBE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  #read README.md to make

# --- SETUP ---
csv_path = 'muse_v3.csv'
model_path = 'model.h5'
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load dataset
df = pd.read_csv(csv_path)
df['link'] = df['lastfm_url']
df['name'] = df['track']
df['emotional'] = df['number_of_emotion_tags']
df['pleasant'] = df['valence_tags']
df['audio_link'] = [None] * len(df)
df = df[['name', 'emotional', 'pleasant', 'link', 'artist', 'audio_link']]
df = df.sort_values(by=["emotional", "pleasant"]).reset_index(drop=True)

# Split by mood
df_sad = df[:18000]
df_angry = df[18000:36000]
df_neutral = df[36000:54000]
df_happy = df[54000:72000]
df_surprised = df[72000:]

filtered_emotions = {"Angry", "Happy", "Neutral", "Sad", "Surprised"}

# Load CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])
model.load_weights(model_path)

emotion_dict = {
    0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy",
    4: "Neutral", 5: "Sad", 6: "Surprised"
}

# Spotify auth
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET))

# --- STREAMLIT UI ---
st.set_page_config(page_title="Emotion Music Recommender", layout="centered")

# Inject custom CSS
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
    }
    h1, h2, h3, h4 {
        color: #ffffff;
    }
    .stButton > button {
        background-color: #1DB954;
        color: white;
        border: none;
        padding: 0.5em 1.5em;
        border-radius: 12px;
        font-weight: bold;
        transition: 0.3s ease-in-out;
    }
    .stButton > button:hover {
        background-color: #1fff96;
        transform: scale(1.05);
    }
    .stRadio > div {
        background-color: #dfecf5;
        padding: 10px;
        border-radius: 12px;
        color: white;
    }
    .stAlert {
        border-radius: 10px;
    }
    audio {
        background-color: transparent;
        border: none;
    }
    a {
        color: #1DB954;
        text-decoration: none;
        font-weight: 600;
    }
    a:hover {
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎵 Emotion-Based Music Recommendation")

# --- FUNCTIONS ---
def clean_emotions(emotion_list):
    seen = set()
    result = []
    for e in emotion_list:
        if e in filtered_emotions and e not in seen:
            result.append(e)
            seen.add(e)
    return result

def get_spotify_preview_and_link(song, artist):
    try:
        q = f"track:{song} artist:{artist}"
        results = sp.search(q=q, type='track', limit=1)
        items = results['tracks']['items']
        if items:
            return items[0]['preview_url'], items[0]['external_urls']['spotify']
    except Exception as e:
        print(f"Spotify API error: {e}")
    return None, None

def get_youtube_video_embed(song, artist):
    search_query = f"{song} {artist} official audio"
    url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={search_query}&key={YOUTUBE_API_KEY}&type=video&maxResults=1"
    res = requests.get(url).json()
    if res.get("items"):
        video_id = res["items"][0]["id"]["videoId"]
        return f"https://www.youtube.com/embed/{video_id}"
    return None

def get_recommendations(emotion_list):
    data = pd.DataFrame()
    emotion_weights = {
        1: [30], 2: [30, 20], 3: [55, 20, 15],
        4: [30, 29, 18, 9], 5: [10, 7, 6, 5, 2]
    }
    selected_weights = emotion_weights.get(len(emotion_list), [10])
    for idx, emotion in enumerate(emotion_list):
        n = selected_weights[idx] if idx < len(selected_weights) else 5
        if emotion == 'Neutral':
            data = pd.concat([data, df_neutral.sample(n=n)], ignore_index=True)
        elif emotion == 'Angry':
            data = pd.concat([data, df_angry.sample(n=n)], ignore_index=True)
        elif emotion == 'Happy':
            data = pd.concat([data, df_happy.sample(n=n)], ignore_index=True)
        elif emotion == 'Surprised':
            data = pd.concat([data, df_surprised.sample(n=n)], ignore_index=True)
        else:
            data = pd.concat([data, df_sad.sample(n=n)], ignore_index=True)

    for i in range(len(data)):
        preview_url, spotify_url = get_spotify_preview_and_link(data.at[i, 'name'], data.at[i, 'artist'])
        data.at[i, 'audio_link'] = preview_url
        data.at[i, 'spotify_url'] = spotify_url
        data.at[i, 'youtube_embed'] = get_youtube_video_embed(data.at[i, 'name'], data.at[i, 'artist'])

    return data

def detect_emotions_from_frame(frame):
    emotions = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi = gray[y:y + h, x:x + w]
        resized_roi = cv2.resize(roi, (48, 48))
        cropped = np.expand_dims(np.expand_dims(resized_roi, -1), 0)
        prediction = model.predict(cropped, verbose=0)
        emotion = emotion_dict[np.argmax(prediction)]
        emotions.append(emotion)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame, emotions

# --- MAIN UI ---
mode = st.radio("Choose a mode:", ["Quick Scan", "Live Mode (Preview + Scan)"])

if mode == "Quick Scan":
    if st.button("Scan Emotion (Quick Mode)"):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Webcam not detected.")
        else:
            st.info("Detecting emotion...")
            emotion_list, count = [], 0
            first_frame = None

            while count < 5:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_with_overlay, emotions = detect_emotions_from_frame(frame)
                emotion_list.extend(emotions)
                if first_frame is None:
                    first_frame = frame_with_overlay.copy()
                count += 1
                time.sleep(0.05)

            cap.release()
            cv2.destroyAllWindows()

            if first_frame is not None:
                st.subheader("📸 Captured Image")
                st.image(first_frame, channels="BGR")

            emotion_list = clean_emotions(emotion_list)
            if emotion_list:
                st.success(f"Detected Emotion(s): {', '.join(emotion_list)}")
                st.subheader("🎶 Recommended Songs")
                recommendations = get_recommendations(emotion_list)
                for i, row in recommendations.head(15).iterrows():
                    if row.get('spotify_url'):
                        st.markdown(f"**{i + 1}. [{row['name']}]({row['spotify_url']})** — *{row['artist']}*")
                    else:
                        st.markdown(f"**{i + 1}. {row['name']}** — *{row['artist']}*")

                    if row['audio_link']:
                        st.audio(row['audio_link'], format="audio/mp3")
                    if row.get('youtube_embed'):
                        st.components.v1.iframe(row['youtube_embed'], height=300)
            else:
                st.warning("No valid emotion detected.")

elif mode == "Live Mode (Preview + Scan)":
    capture_flag = st.button("Capture Emotion (from Live Feed)")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Webcam not detected.")
    else:
        stframe = st.empty()
        captured_frame = None
        all_emotions = []
        st.info("Live feed active... Click the capture button to analyze.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_with_overlay, emotions = detect_emotions_from_frame(frame)
            stframe.image(frame_with_overlay, channels="BGR", use_column_width=True)

            if capture_flag:
                captured_frame = frame_with_overlay.copy()
                all_emotions.extend(emotions)
                break

        cap.release()
        cv2.destroyAllWindows()

        if captured_frame is not None:
            st.subheader("📸 Captured Frame")
            st.image(captured_frame, channels="BGR")
            emotion_list = clean_emotions(all_emotions)
            if emotion_list:
                st.success(f"Detected Emotion(s): {', '.join(emotion_list)}")
                st.subheader("🎶 Recommended Songs")
                recommendations = get_recommendations(emotion_list)
                for i, row in recommendations.head(15).iterrows():
                    if row.get('spotify_url'):
                        st.markdown(f"**{i + 1}. [{row['name']}]({row['spotify_url']})** — *{row['artist']}*")
                    else:
                        st.markdown(f"**{i + 1}. {row['name']}** — *{row['artist']}*")

                    if row['audio_link']:
                        st.audio(row['audio_link'], format="audio/mp3")
                    if row.get('youtube_embed'):
                        st.components.v1.iframe(row['youtube_embed'], height=300)
            else:
                st.warning("No valid emotion detected.")
