
import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FILE = "final_book_liked_percentage_model.pkl"
DATA_FILE = "final_book_deployment_data.csv"

FEATURES = [
    "genres",
    "rating",
    "numofratings",
    "bbescores",
    "pages",
    "price"
]


# ============================================================
# LOAD MODEL AND DATA
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)


# ============================================================
# BOOK SEARCH
# ============================================================

def search_books(book_name, data):

    search_text = str(book_name).strip().lower()

    if not search_text:
        return data.iloc[0:0]

    return data[
        data["title"]
        .astype(str)
        .str.lower()
        .str.contains(
            search_text,
            na=False,
            regex=False
        )
    ]


# ============================================================
# GENRE FUNCTIONS
# ============================================================

def get_genres(data):

    genres = set()

    for value in data["genres"].dropna():

        for genre in str(value).split(","):

            genre = genre.strip()

            if genre:
                genres.add(genre)

    return sorted(genres)


def get_books_by_genre(data, selected_genre):

    genre_text = (
        data["genres"]
        .fillna("")
        .astype(str)
    )

    return data[
        genre_text.str.contains(
            selected_genre,
            case=False,
            na=False,
            regex=False
        )
    ].copy()


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_book(book_row, model, data):

    input_data = pd.DataFrame([{
        "genres": book_row["genres"],
        "rating": book_row["rating"],
        "numofratings": book_row["numofratings"],
        "bbescores": book_row["bbescores"],
        "pages": book_row["pages"],
        "price": book_row["price"]
    }])

    # Handle missing genre
    input_data["genres"] = (
        input_data["genres"]
        .fillna("Unknown")
        .astype(str)
    )

    # Handle numeric features
    numeric_features = [
        "rating",
        "numofratings",
        "bbescores",
        "pages",
        "price"
    ]

    for feature in numeric_features:

        input_data[feature] = pd.to_numeric(
            input_data[feature],
            errors="coerce"
        )

        if pd.isna(input_data.loc[0, feature]):

            input_data.loc[0, feature] = (
                pd.to_numeric(
                    data[feature],
                    errors="coerce"
                ).median()
            )

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Keep prediction within valid percentage range
    prediction = np.clip(
        prediction,
        0,
        100
    )

    return round(float(prediction), 2)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Book Liked Percentage Predictor",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# LOAD APPLICATION RESOURCES
# ============================================================

try:

    model = load_model()
    data = load_data()

except Exception as e:

    st.error(
        f"Unable to load model or dataset: {e}"
    )

    st.stop()


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("📚 Book Liked Percentage Predictor")

st.write(
    "Predict the liked percentage of a book using "
    "the trained machine learning model."
)

st.write(
    f"Dataset contains **{len(data):,} books**."
)


# ============================================================
# SIDEBAR - BROWSE BY GENRE
# ============================================================

st.sidebar.title("📚 Browse Books")

st.sidebar.write(
    "Select a genre and then choose a book."
)

genres = get_genres(data)

selected_genre = st.sidebar.selectbox(
    "Select Genre",
    ["-- Choose a genre --"] + genres
)


if selected_genre != "-- Choose a genre --":

    genre_books = get_books_by_genre(
        data,
        selected_genre
    )

    st.sidebar.write(
        f"{len(genre_books):,} books found."
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
        ["-- Choose a book --"] + book_options
    )

    if selected_book_option != "-- Choose a book --":

        selected_index = (
            book_options.index(
                selected_book_option
            )
        )

        selected_book = genre_books.iloc[
            selected_index
        ]

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

        if st.button(
            "Predict Liked Percentage",
            key="genre_prediction"
        ):

            prediction = predict_book(
                selected_book,
                model,
                data
            )

            st.success(
                f"Predicted Liked Percentage: "
                f"{prediction:.2f}%"
            )


# ============================================================
# DIRECT BOOK SEARCH
# ============================================================

st.divider()

st.subheader("🔎 Search for a Book")

book_name = st.text_input(
    "Enter book name",
    placeholder="Example: The Golem and the Jinni"
)


if book_name:

    results = search_books(
        book_name,
        data
    )

    if len(results) == 0:

        st.warning(
            "No matching book found in the dataset."
        )

    elif len(results) == 1:

        selected_book = results.iloc[0]

        prediction = predict_book(
            selected_book,
            model,
            data
        )

        st.write(
            f"**Book:** {selected_book['title']}"
        )

        st.write(
            f"**Author:** {selected_book['author']}"
        )

        st.write(
            f"**Rating:** {selected_book['rating']}"
        )

        st.write(
            f"**Number of Ratings:** "
            f"{selected_book['numofratings']}"
        )

        st.success(
            f"Predicted Liked Percentage: "
            f"{prediction:.2f}%"
        )

    else:

        st.info(
            f"{len(results)} matching books found. "
            "Please select one."
        )

        options = []

        for _, row in results.iterrows():

            title = str(row["title"])
            author = str(row["author"])

            options.append(
                f"{title} — {author}"
            )

        selected_option = st.selectbox(
            "Select a book",
            options,
            key="search_book_selection"
        )

        selected_index = options.index(
            selected_option
        )

        selected_book = results.iloc[
            selected_index
        ]

        if st.button(
            "Predict Liked Percentage",
            key="search_prediction"
        ):

            prediction = predict_book(
                selected_book,
                model,
                data
            )

            st.write(
                f"**Book:** {selected_book['title']}"
            )

            st.write(
                f"**Author:** {selected_book['author']}"
            )

            st.success(
                f"Predicted Liked Percentage: "
                f"{prediction:.2f}%"
            )
