import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("movies_metadata.csv", low_memory=False)
    df = df[['title', 'overview']].dropna()
    return df

df = load_data()

# -----------------------------
# Build TF-IDF Matrix
# -----------------------------
@st.cache_resource
def build_model(dataframe):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(dataframe['overview'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = build_model(df)

# -----------------------------
# Recommendation Function
# -----------------------------
def recommend(movie_title):
    matches = df[df['title'].str.lower() == movie_title.lower()]
    if matches.empty:
        return ["❌ Movie not found."]
    
    idx = matches.index[0]
    scores = list(enumerate(cosine_sim[idx]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    recommendations = []
    for i in sorted_scores[1:6]:
        recommendations.append(df.iloc[i[0]].title)
    return recommendations

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🎬 Movie Recommendation System")
st.write("Get similar movie recommendations based on the overview content!")

movie_title = st.text_input("Enter a movie title (e.g. The Dark Knight):")

if st.button("Recommend"):
    with st.spinner("Finding similar movies..."):
        results = recommend(movie_title)
        st.subheader("Recommended Movies:")
        for movie in results:
            st.write(f"🎥 {movie}")
