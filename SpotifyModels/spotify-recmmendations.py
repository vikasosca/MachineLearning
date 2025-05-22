import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib

def recommend_songs(track_name, n=10):

    #Load the file
    #df = pd.read_csv(("C:/Users/Admin/Desktop/AI/python/Datasets/SpotifyFeatures.csv"))
    # Load everything back
    df = joblib.load('spotify_df.pkl')
    features_scaled = joblib.load('features_scaled.pkl')
    scaler = joblib.load('scaler.pkl')
    
    feature_cols = ["acousticness"	,"danceability"	,"energy","instrumentalness"	
                    ,"liveness"	,"speechiness"	,"tempo", "valence"]
    features = df.dropna(subset = feature_cols)
    features = df[feature_cols]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Pick a song by name (you can change this)
    seed_song = track_name
    seed = df[df['track_name'].str.lower() == seed_song.lower()].iloc[0]
    seed_vector = features_scaled[seed.name]

    # Compute similarity between the seed song and all others
    similarities = cosine_similarity([seed_vector],features_scaled)[0]
    df['similarity'] = similarities
    # Sort by similarity (excluding the seed song itself)
    recommendations = df[df['track_name'].str.lower() != seed_song.lower()] \
                    .sort_values(by='similarity', ascending=False)

    #joblib.dump(df, 'spotify_df.pkl')

    # Save the scaled feature array
    #joblib.dump(features_scaled, 'features_scaled.pkl')

    # Save the fitted scaler
    #joblib.dump(scaler, 'scaler.pkl')
    print("Top 10 recommended songs similar to:", seed_song)
    return recommendations[['track_name', 'similarity']].head(n)
   
    
print(recommend_songs("Catfish Blues", n=10))