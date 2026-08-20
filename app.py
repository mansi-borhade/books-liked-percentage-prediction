
import streamlit as st
import pandas as pd
import numpy as np
import joblib

MODEL_FILE = "final_book_liked_percentage_model.pkl"
DATA_FILE = "final_book_deployment_data.csv"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)


def get_genres(data):
    genres = set()

    for value in data["genres"].dropna():
        for genre in str(value).split(","):
            genre = genre.strip()

            if genre:
                genres.add(genre)

    return sorted(genres)


def get_books_by_genre(data, selected_genre):
    mask = data["genres"].fillna("").astype(str).str.contains(
        selected_genre,
        case=False,
        na=False,
        regex=False
    )

    return data[mask].copy()


def predict_book(book_row, model):
    input_data = pd.DataFrame([{
        "genres": book_row["genres"],
        "rating": book_row["rating"],
        "numofratings": book_row["numofratings"],
        "bbescores": book_row["bbescores"],
        "pages": book_row["pages"],
        "price": book_row["price"]
    }])

    prediction = model.predict(input_data)[0]

    prediction = np.clip(prediction, 0, 100)

    return round(float(prediction), 2)


st.set_page_config(
    page_title="Book Liked Percentage Predictor",
    page_icon="📚",
    layout="wide"
)


model = load_model()
data = load_data()


st.title("📚 Book Liked Percentage Predictor")

st.write(
    "Select a genre and then choose a book to predict "
    "its liked percentage."
)


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

st.sidebar.title("📚 Browse Books")

genres = get_genres(data)

selected_genre = st.sidebar.selectbox(
    "Select Genre",
    genres
)


genre_books = get_books_by_genre(
    data,
    selected_genre
)


st.sidebar.write(
    f"{len(genre_books)} books found in this genre."
)


book_options = []

for _, row in genre_books.iterrows():

    title = str(row["title"])
    author = str(row["author"])

    book_options.append(
        f"{title} — {author}"
    )


selected_book_option = st.sidebar.selectbox(
    "Select Book",
    book_options
)


selected_index = book_options.index(
    selected_book_option
)


selected_book = genre_books.iloc[selected_index]


# ------------------------------------------------------------
# MAIN PAGE
# ------------------------------------------------------------

st.subheader("Selected Book")

st.write(
    f"**Book:** {selected_book['title']}"
)

st.write(
    f"**Author:** {selected_book['author']}"
)

st.write(
    f"**Genre:** {selected_book['genres']}"
)


if st.button("Predict Liked Percentage"):

    prediction = predict_book(
        selected_book,
        model
    )

    st.success(
        f"Predicted Liked Percentage: {prediction:.2f}%"
    )
