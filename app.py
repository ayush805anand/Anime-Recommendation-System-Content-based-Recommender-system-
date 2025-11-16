import streamlit as st
import pickle
import numpy as np

anime = pickle.load(open("anime_list.pkl", "rb"))
similarity = pickle.load(open("similarity.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
scaler = pickle.load(open("num_scaler.pkl", "rb"))

if "rating" in anime.columns:
    anime["rating_original"] = anime["rating"]
elif "rating_trans" in anime.columns:
    anime["rating_original"] = anime["rating_trans"]
else:
    anime["rating_original"] = np.nan

anime["type"] = anime["type"].fillna("").replace("", "Not Applicable / NA")

def filter_by_category(df, category):
    if category == "All":
        return df
    return df[df["type"].str.lower() == category.lower()]

def recommend(anime_name, category):
    df_filtered = filter_by_category(anime, category)
    match = anime[anime["name"].str.lower() == anime_name.lower()]
    if match.empty:
        return []
    idx = match.index[0]
    distances = list(enumerate(similarity[idx]))
    valid_indices = df_filtered.index.tolist()
    distances = [
        (i, d) for i, d in distances
        if i in valid_indices and (d < 0.999999 or i == idx)
    ]
    distances = sorted(distances, key=lambda x: x[1], reverse=True)
    distances = [x for x in distances if x[0] != idx][:3]

    recommendations = []
    for ind, score in distances:
        row = anime.loc[ind]
        recommendations.append({
            "name": row["name"],
            "type": row["type"],
            "rating_original": row["rating_original"],
            "similarity_score": round(score, 3)
        })

    recommendations = sorted(
        recommendations,
        key=lambda x: x["rating_original"],
        reverse=True
    )

    return recommendations

st.set_page_config(page_title="Anime Recommender", layout="wide")
st.image("bg.jpg", use_container_width=True)
st.title("Anime Recommendation System")
st.write("Content-Based Recommender: Choose from over 12,000 anime titles!")
anime_names_sorted = sorted(anime["name"].unique())
default_name = "Shigatsu wa Kimi no Uso"
default_index = anime_names_sorted.index(default_name) if default_name in anime_names_sorted else 0
selected_anime = st.selectbox(
    "Choose or type the anime name:",
    anime_names_sorted,
    index=default_index
)

categories = ["All"] + sorted(anime["type"].dropna().unique().tolist())
selected_category = st.selectbox("Select Anime Category:", categories)

if st.button("Recommend"):
    results = recommend(selected_anime, selected_category)
    if len(results) == 0:
        st.error("No matching recommendations found for this category.")
    else:
        st.write(f"Top 3 Recommendations for {selected_anime} ({selected_category})")

        for i, rec in enumerate(results, start=1):
            st.markdown(
                f"""
                {i}. {rec['name']}  
                - Type: {rec['type']}  
                - Rating: {rec['rating_original']:.2f}  
                - Similarity: {rec['similarity_score']}
                """,
                unsafe_allow_html=True
            )
st.write("---")
st.write("Built using Streamlit by Ayush Anand.")
