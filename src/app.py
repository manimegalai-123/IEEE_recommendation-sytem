
import streamlit as st
import pandas as pd
import html

import recommend as recommender


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Personalized Movie Recommender",
    page_icon="🎬",
    layout="wide"
)


# ==========================================
# CUSTOM STYLING
# ==========================================

st.markdown("""
<style>
.stApp {
    background-color: #10111b;
    color: white;
}

h1, h2, h3 {
    color: #ffffff;
}

.movie-card {
    background-color: #202333;
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 12px;
    border: 1px solid #34384e;
}

.movie-title {
    color: #ffffff;
    font-size: 20px;
    font-weight: bold;
}

.movie-info {
    color: #c5c8d8;
    font-size: 14px;
}

.score {
    color: #55d6a5;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# HEADER
# ==========================================

st.title("🎬 Personalized Movie Recommendation System")

st.markdown(
    "Discover movies based on your interests and viewing preferences."
)

st.divider()


# ==========================================
# USER INPUT
# ==========================================

st.subheader("👤 Find Your Recommendations")

user_id = st.number_input(
    "Enter your MovieLens User ID",
    min_value=1,
    value=10,
    step=1
)

top_k = st.selectbox(
    "Number of recommendations",
    [5, 10, 15, 20],
    index=1
)


# ==========================================
# GENERATE RECOMMENDATIONS
# ==========================================

if st.button("✨ Get My Recommendations", use_container_width=True):

    original_user_id = int(user_id)

    if original_user_id not in recommender.user_to_index:

        st.error(
            "User ID not found. Please enter a valid MovieLens User ID."
        )

    else:

        internal_id = recommender.user_to_index[original_user_id]

        st.success("Your personalized recommendations are ready!")

        # ======================================
        # USER HISTORY
        # ======================================

        st.subheader("❤️ Your Previously Liked Movies")

        history = recommender.train[
            recommender.train[:, 0] == internal_id
        ]

        history_item_ids = history[:, 1].astype(int)

        # Retrieve movie details safely
        history_movies = (
            recommender.aligned_movies
            .iloc[history_item_ids][["title", "genres"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        st.metric(
            "Previously liked movies",
            len(history_movies)
        )

        if not history_movies.empty:

            for start in range(0, min(len(history_movies), 10), 2):

                cols = st.columns(2)

                for col, movie_index in zip(
                    cols,
                    range(start, min(start + 2, min(len(history_movies), 10)))
                ):

                    movie = history_movies.iloc[movie_index]

                    with col:

                        title = html.escape(str(movie["title"]))
                        genres = html.escape(str(movie["genres"]))

                        st.markdown(
                            f"""
                            <div class="movie-card">
                                <div class="movie-title">
                                    🎞️ {title}
                                </div>
                                <p class="movie-info">
                                    Genres: {genres}
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

        else:

            st.info("No previous movie history found.")

        # ======================================
        # RECOMMENDATIONS
        # ======================================

        st.subheader("🍿 Recommended Movies For You")

        recommendations = recommender.recommend_movies(
            original_user_id=original_user_id,
            top_k=int(top_k)
        )

        if recommendations is not None and not recommendations.empty:

            recommendations = recommendations.reset_index(drop=True)

            st.write(
                f"Here are your top {len(recommendations)} recommendations."
            )

            for index, movie in recommendations.iterrows():

                title = html.escape(str(movie["title"]))
                genres = html.escape(str(movie["genres"]))
                score = float(movie["Hybrid Score"])

                st.markdown(
                    f"""
                    <div class="movie-card">
                        <div class="movie-title">
                            {index + 1}. {title}
                        </div>

                        <p class="movie-info">
                            <b>Genres:</b> {genres}
                        </p>

                        <p class="score">
                            Hybrid Score: {score:.4f}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:

            st.warning("No recommendations could be generated.")


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Powered by BPR Matrix Factorization + TF-IDF Hybrid Recommendation."
)

st.caption(
    "Dataset: MovieLens 25M | Personalized recommendation prototype"
)
