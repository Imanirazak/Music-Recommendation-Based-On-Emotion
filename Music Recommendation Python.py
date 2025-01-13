import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import lightgbm
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
import tkinter as tk
from tkinter import messagebox
import random
import os

# Set environment variable to avoid subprocess warnings
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Load and preprocess data
data = pd.read_csv('data.csv.zip', compression='zip')
data.drop_duplicates(inplace=True, subset=['name'])

name = data['name']

# Clustering
col_features = ['danceability', 'energy', 'valence', 'loudness']
X = MinMaxScaler().fit_transform(data[col_features])
kmeans = KMeans(init="k-means++", n_clusters=2, random_state=15).fit(X)
data['kmeans'] = kmeans.labels_
data['song_name'] = name

# Group by clusters
cluster = data.groupby(by=data['kmeans'])

y = data.pop('kmeans')
x = data.drop(columns=['name', 'artists', 'id', 'release_date', 'song_name'])

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Model training
model = LGBMClassifier(verbose=-1).fit(x_train, y_train)

# Feature importance plot
ax = lightgbm.plot_importance(model, max_num_features=10, figsize=(15, 15))

# Update clustering with new settings
kmeans = KMeans(n_clusters=3)
data['kmeans'] = kmeans.fit_predict(data[['energy', 'danceability', 'valence']])

# Apply groupby and sort by 'popularity' within each group
cluster = data.groupby('kmeans')
df = cluster.apply(lambda x: x.sort_values(by="popularity", ascending=False)[x.columns])
df.reset_index(level=0, drop=True, inplace=True)

# Emotion detection setup
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'final_model.h5'
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["happy", "sad"]

# Emotion detection function
def emotion_testing():
    cap = cv2.VideoCapture(0)
    predicted_emotion = None

    while True:
        ret, test_img = cap.read()
        if not ret:
            continue

        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        faces_detected = face_detection.detectMultiScale(gray_img, 1.32, 5)

        for (x, y, w, h) in faces_detected:
            cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
            roi_gray = gray_img[y:y + w, x:x + h]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255

            predictions = emotion_classifier.predict(img_pixels)

            max_index = np.argmax(predictions[0])
            predicted_emotion = EMOTIONS[max_index]

            cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        resized_img = cv2.resize(test_img, (1000, 700))
        cv2.imshow('Facial Emotion Analysis', resized_img)

        if predicted_emotion in ['happy', 'sad']:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return predicted_emotion

# Song suggestion popup
def suggest_song(song_name):
    popup = tk.Tk()
    popup.title("Song Recommendation")
    popup.geometry("300x150")

    label = tk.Label(popup, text=f"Emotion Detected:{emotion_word}\n\nWould you like to listen to \n'{song_name}'?", wraplength=280, justify="center")
    label.pack(pady=20)

    def on_yes():
        print(f"You chose to listen to {song_name}")
        popup.destroy()

    def on_no():
        print(f"You chose not to listen to {song_name}")
        popup.destroy()

    yes_button = tk.Button(popup, text="Yes", command=on_yes, width=10)
    no_button = tk.Button(popup, text="No", command=on_no, width=10)

    yes_button.pack(side="left", padx=20, pady=10)
    no_button.pack(side="right", padx=20, pady=10)

    popup.mainloop()

# Keep track of suggested songs for each emotion
suggested_songs_history = {"happy": set(), "sad": set()}

# Get results with unique suggestions
def get_results(emotion_code):
    NUM_RECOMMEND = 10
    emotion_label = "sad" if emotion_code == 0 else "happy"
    cluster_label = 0 if emotion_code == 0 else 1

    # Get the songs from the corresponding cluster
    song_set = df[df['kmeans'] == cluster_label]['song_name']

    # Convert to a list and shuffle it for randomness
    song_list = list(song_set)
    random.shuffle(song_list)

    # Filter out already suggested songs
    song_list = [song for song in song_list if song not in suggested_songs_history[emotion_label]]

    # If all songs are suggested, reset the history for this emotion
    if not song_list:
        suggested_songs_history[emotion_label] = set()
        song_list = list(song_set)
        random.shuffle(song_list)

    # Pick top NUM_RECOMMEND songs and update the history
    recommended_songs = song_list[:NUM_RECOMMEND]
    suggested_songs_history[emotion_label].update(recommended_songs)

    # Return the songs as a DataFrame for compatibility
    return pd.Series(recommended_songs).reset_index(drop=True)

# Main execution
emotion_word = emotion_testing()
if emotion_word == 'sad':
    emotion_code = 0
else:
    emotion_code = 1

songs = get_results(emotion_code)

print(songs)

top_song = songs.iloc[0] if not songs.empty else "No song available"
suggest_song(top_song)
