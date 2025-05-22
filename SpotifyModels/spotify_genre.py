import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

'''
1. Prepare the Data
2. Train/Test Split
3. Preprocess (StandardScaler)
4. Train a Classifier (e.g., Logistic Regression or RandomForestClassifier)
5. Evaluate the Model
'''
def find_song_genre(larray):
    
    df = pd.read_csv(("C:/Users/Admin/Desktop/AI/python/Datasets/SpotifyFeatures.csv"))
    # Filter to top 8 genres
    top_genres = df['genre'].value_counts().nlargest(8).index
    df = df[df['genre'].isin(top_genres)]
    df_genre = df['genre']
    features = ['acousticness',	'danceability','energy','instrumentalness','liveness','speechiness','tempo','valence']
    df = df.dropna(subset=features)
    feature_cols = df[features]
    genre = df_genre

    scaler = StandardScaler()

    X_train,X_test,y_train,y_test = train_test_split(feature_cols,genre,test_size=0.2,random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    clf.fit(X_train_scaled,y_train)

    y_pred = clf.predict(X_test_scaled)
    #Accuracy of the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    #Confusion Matrix
    cm = confusion_matrix(y_test,y_pred,labels =clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot(xticks_rotation=45)

    # Visualize features
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_test_scaled)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=pd.factorize(y_test)[0], cmap='viridis', alpha=0.5)
    plt.legend(*scatter.legend_elements(), title="Genres")
    plt.title("PCA Projection of Test Set by Genre")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.show()
    '''
    #  Test for a single song
    new_song = np.array(larray)
    new_song_scaler = scaler.transform(new_song)
    y_pred = clf.predict(new_song_scaler)
    #Accuracy of the model
    print("----------------------------------------------------------")
    print("Predicted Genre of new song:", y_pred[0])
    '''
    joblib.dump(clf, 'spotify_genre_model.pkl')
    joblib.dump(scaler, 'spotify_scaler.pkl')

def predict_genre(features, model, scaler):
    #  Test for a single song
    feature_scaled = scaler.transform(features)
    prediction = model.predict(feature_scaled)   
    #Accuracy of the model
    print("----------------------------------------------------------")
    print("Predicted Genre of this song:", prediction[0])
    return prediction[0]

my_song = np.array([[0.5, 0.25, 0.133, 0.1729, 0.054, 0.035, 100.0, 0.2]])  # shape: (1, 8)
model = joblib.load('spotify_genre_model.pkl')
scaler = joblib.load('spotify_scaler.pkl')
print(predict_genre(my_song, model, scaler)) 