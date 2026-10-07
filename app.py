import html
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import load_npz


# ============================================================
# RECOMAI
# AI-Powered Beauty Product Recommendation Engine
# ============================================================

st.set_page_config(
    page_title="RecomAI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# THEME
# ============================================================

st.html("""
<style>

:root {
    --bg: #0F1020;
    --surface: #17182B;
    --card: #1D1E33;
    --hover: #242541;
    --border: #2D2E48;
    --primary: #A78BFA;
    --light: #C4B5FD;
    --coral: #FB7185;
    --text: #F8FAFC;
    --muted: #A1A1B5;
}

.stApp {
    background: var(--bg);
    color: var(--text);
}

[data-testid="stHeader"] {
    background: rgba(15, 16, 32, 0.94);
}

[data-testid="stSidebar"] {
    background: #121326;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: var(--text);
}

.hero {
    background:
        radial-gradient(
            circle at 90% 10%,
            rgba(251, 113, 133, 0.12),
            transparent 32%
        ),
        radial-gradient(
            circle at 10% 0%,
            rgba(167, 139, 250, 0.14),
            transparent 34%
        ),
        var(--surface);

    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 30px;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 34px;
    margin: 0 0 8px;
    color: var(--text);
}

.hero p {
    color: var(--muted);
    font-size: 16px;
    line-height: 1.6;
    max-width: 850px;
    margin: 0;
}

.section-title {
    font-size: 25px;
    font-weight: 750;
    color: var(--text);
    margin: 25px 0 14px;
}

.section-subtitle {
    color: var(--muted);
    font-size: 14px;
    margin: -8px 0 18px;
}

.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 18px;
    min-height: 105px;
}

.metric-label {
    color: var(--muted);
    font-size: 13px;
    margin-bottom: 6px;
}

.metric-value {
    color: var(--text);
    font-size: 25px;
    font-weight: 750;
}

.model-box {
    background:
        linear-gradient(
            135deg,
            rgba(167, 139, 250, 0.11),
            rgba(251, 113, 133, 0.06)
        );

    border: 1px solid rgba(167, 139, 250, 0.28);
    border-radius: 16px;
    padding: 19px;
    margin: 15px 0 22px;
}

.model-title {
    font-size: 19px;
    font-weight: 700;
    color: var(--light);
    margin-bottom: 5px;
}

.model-description {
    color: var(--muted);
    font-size: 14px;
    line-height: 1.55;
}

.product-title {
    color: var(--text);
    font-size: 15px;
    font-weight: 650;
    line-height: 1.4;
    min-height: 63px;
    margin-top: 9px;
}

.product-meta {
    color: var(--muted);
    font-size: 13px;
    margin-top: 7px;
}

.price {
    color: var(--light);
    font-size: 17px;
    font-weight: 750;
    margin-top: 7px;
}

.rank-badge {
    display: inline-block;
    background: rgba(167, 139, 250, 0.14);
    color: var(--light);
    border: 1px solid rgba(167, 139, 250, 0.25);
    border-radius: 8px;
    padding: 4px 9px;
    font-size: 12px;
    font-weight: 700;
    margin-bottom: 10px;
}

.match {
    display: inline-block;
    background: rgba(251, 113, 133, 0.12);
    color: #FDA4AF;
    border: 1px solid rgba(251, 113, 133, 0.25);
    border-radius: 8px;
    padding: 5px 9px;
    font-size: 12px;
    font-weight: 700;
    margin-top: 8px;
}

.explain {
    background: rgba(167, 139, 250, 0.06);
    border-left: 3px solid var(--primary);
    padding: 8px 10px;
    border-radius: 5px;
    color: var(--muted);
    font-size: 12px;
    margin-top: 10px;
}

.footer {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px;
    text-align: center;
    color: var(--muted);
    margin-top: 30px;
}

</style>
""")


# ============================================================
# LOAD SAVED MODELS
# ============================================================

@st.cache_resource
def load_models():

    svd_model = joblib.load(
        "models/svd_model.pkl"
    )

    knn_model = joblib.load(
        "models/knn_model.pkl"
    )

    item_similarity = load_npz(
        "models/item_similarity.npz"
    )

    item_user_matrix = load_npz(
        "models/item_user_matrix.npz"
    )

    mappings = joblib.load(
        "models/item_cf_mappings.pkl"
    )

    product_catalog = pd.read_pickle(
        "models/product_catalog.pkl"
    )

    product_lookup = (
        product_catalog
        .drop_duplicates("parent_asin")
        .set_index("parent_asin")
    )

    return (
        svd_model,
        knn_model,
        item_similarity,
        item_user_matrix,
        mappings,
        product_catalog,
        product_lookup
    )


(
    svd_model,
    knn_model,
    item_similarity,
    item_user_matrix,
    mappings,
    product_catalog,
    product_lookup
) = load_models()


# ============================================================
# HELPERS
# ============================================================

def clean_image_url(url):

    if not isinstance(url, str):
        return None

    if "](" in url:
        url = url.split("](", 1)[1]
        url = url.rstrip(")")

    return url


def get_product(product_id):

    try:

        product = product_lookup.loc[
            product_id
        ]

        if isinstance(product, pd.DataFrame):
            return product.iloc[0]

        return product

    except KeyError:

        return None


def normalize_scores(scores):

    if not scores:
        return {}

    values = np.asarray(
        list(scores.values()),
        dtype=float
    )

    minimum = values.min()
    maximum = values.max()

    if maximum == minimum:

        return {
            key: 1.0
            for key in scores
        }

    return {
        key:
        (value - minimum) /
        (maximum - minimum)

        for key, value in scores.items()
    }


def create_demo_users():

    result = []

    for inner_id in range(
        svd_model.trainset.n_users
    ):

        user_id = (
            svd_model
            .trainset
            .to_raw_uid(inner_id)
        )

        interaction_count = len(
            svd_model
            .trainset
            .ur[inner_id]
        )

        if interaction_count >= 2:

            result.append(
                {
                    "user_id": user_id,
                    "interactions": interaction_count
                }
            )

        if len(result) >= 10:
            break

    return result


demo_users = create_demo_users()


# ============================================================
# USER HISTORY
# ============================================================

def get_user_history(user_id):

    user_index = mappings[
        "user_to_index"
    ].get(user_id)

    if user_index is None:
        return None, []

    product_indices = (
        item_user_matrix[:, user_index]
        .nonzero()[0]
    )

    history = []

    for product_index in product_indices:

        product_id = mappings[
            "products"
        ][product_index]

        rating = float(
            item_user_matrix[
                product_index,
                user_index
            ]
        )

        history.append(
            (
                product_id,
                rating
            )
        )

    return user_index, history


# ============================================================
# SVD
# ============================================================

def recommend_svd(
    user_id,
    top_n=5
):

    try:

        user_index = (
            svd_model
            .trainset
            .to_inner_uid(user_id)
        )

    except ValueError:

        return pd.DataFrame()

    user_vector = (
        svd_model
        .pu[user_index]
    )

    scores = (
        svd_model.qi @ user_vector
        + svd_model.bu[user_index]
        + svd_model.bi
        + svd_model.trainset.global_mean
    )

    rated_indices = {
        item_index

        for item_index, _
        in svd_model
        .trainset
        .ur[user_index]
    }

    for item_index in rated_indices:

        scores[item_index] = -np.inf

    top_indices = (
        scores
        .argsort()[-top_n:][::-1]
    )

    return pd.DataFrame(
        {
            "parent_asin": [
                svd_model
                .trainset
                .to_raw_iid(index)

                for index in top_indices
            ],

            "predicted_rating": [
                min(
                    5,
                    max(
                        1,
                        float(scores[index])
                    )
                )

                for index in top_indices
            ]
        }
    )


# ============================================================
# KNN / ITEM-CF SHARED LOGIC
# ============================================================

def _neighbor_scores(
    user_id,
    use_precomputed=True,
    top_n=5
):

    _, history = get_user_history(
        user_id
    )

    if not history:
        return pd.DataFrame()

    seen_products = {
        product_id
        for product_id, _
        in history
    }

    scores = {}

    for product_id, rating in history:

        product_index = mappings[
            "product_to_index"
        ].get(product_id)

        if product_index is None:
            continue

        if use_precomputed:

            similarities = (
                item_similarity
                .getrow(product_index)
            )

            pairs = zip(
                similarities.indices,
                similarities.data
            )

        else:

            distances, indices = (
                knn_model.kneighbors(
                    item_user_matrix[
                        product_index
                    ],
                    n_neighbors=min(
                        51,
                        item_user_matrix.shape[0]
                    )
                )
            )

            pairs = (
                (
                    index,
                    1 - distance
                )

                for distance, index
                in zip(
                    distances[0],
                    indices[0]
                )
            )

        for index, similarity in pairs:

            recommended_product = (
                mappings["products"][index]
            )

            if recommended_product in seen_products:
                continue

            if similarity <= 0:
                continue

            scores[
                recommended_product
            ] = (
                scores.get(
                    recommended_product,
                    0
                )
                + float(similarity) * rating
            )

    if not scores:
        return pd.DataFrame()

    recommendations = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    return pd.DataFrame(
        recommendations,
        columns=[
            "parent_asin",
            "score"
        ]
    )


def recommend_item_cf(
    user_id,
    top_n=5
):

    return _neighbor_scores(
        user_id,
        True,
        top_n
    )


def recommend_knn(
    user_id,
    top_n=5
):

    return _neighbor_scores(
        user_id,
        False,
        top_n
    )


# ============================================================
# POPULARITY
# ============================================================

@st.cache_data
def popularity_scores():

    popularity = (
        product_catalog[
            [
                "parent_asin",
                "average_rating",
                "rating_number"
            ]
        ]
        .dropna()
        .copy()
    )

    if popularity.empty:
        return {}

    global_mean = (
        popularity["average_rating"]
        .mean()
    )

    m = 10

    popularity["score"] = (
        (
            popularity["rating_number"]
            /
            (
                popularity["rating_number"]
                + m
            )
        )
        * popularity["average_rating"]

        +

        (
            m
            /
            (
                popularity["rating_number"]
                + m
            )
        )
        * global_mean
    )

    return dict(
        zip(
            popularity["parent_asin"],
            popularity["score"]
        )
    )


# ============================================================
# HYBRID
# ============================================================

def recommend_hybrid(
    user_id,
    top_n=5
):

    user_index, _ = (
        get_user_history(user_id)
    )

    if user_index is None:
        return pd.DataFrame()

    candidate_count = max(
        50,
        top_n * 10
    )

    svd = recommend_svd(
        user_id,
        candidate_count
    )

    item_cf = recommend_item_cf(
        user_id,
        candidate_count
    )

    popularity = normalize_scores(
        popularity_scores()
    )

    svd_scores = {}

    if not svd.empty:

        svd_scores = dict(
            zip(
                svd["parent_asin"],
                svd["predicted_rating"]
            )
        )

    item_cf_scores = {}

    if not item_cf.empty:

        item_cf_scores = dict(
            zip(
                item_cf["parent_asin"],
                item_cf["score"]
            )
        )

    svd_scores = normalize_scores(
        svd_scores
    )

    item_cf_scores = normalize_scores(
        item_cf_scores
    )

    candidates = (
        set(svd_scores)
        |
        set(item_cf_scores)
    )

    top_popular = sorted(
        popularity.items(),
        key=lambda x: x[1],
        reverse=True
    )[:1000]

    candidates.update(
        product_id
        for product_id, _
        in top_popular
    )

    _, history = get_user_history(
        user_id
    )

    seen_products = {
        product_id
        for product_id, _
        in history
    }

    candidates -= seen_products

    hybrid_scores = {}

    for product_id in candidates:

        svd_score = svd_scores.get(
            product_id,
            0
        )

        item_cf_score = item_cf_scores.get(
            product_id,
            0
        )

        popularity_score = popularity.get(
            product_id,
            0
        )

        hybrid_scores[product_id] = (
            0.50 * svd_score
            +
            0.30 * item_cf_score
            +
            0.20 * popularity_score
        )

    if not hybrid_scores:
        return pd.DataFrame()

    recommendations = sorted(
        hybrid_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    return pd.DataFrame(
        recommendations,
        columns=[
            "parent_asin",
            "hybrid_score"
        ]
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

MODEL_INFO = {

    "SVD": (
        "🎯",
        "Learns latent user and product factors from "
        "historical ratings to predict products a user may prefer."
    ),

    "KNN": (
        "🔎",
        "Finds neighboring products using cosine similarity "
        "between product rating patterns."
    ),

    "Item-CF": (
        "🔗",
        "Recommends products based on their similarity "
        "to products the user has already interacted with."
    ),

    "Hybrid": (
        "🔥",
        "Combines SVD, Item-CF and popularity signals "
        "into one recommendation ranking."
    )
}


# ============================================================
# PRODUCT CARD
# ============================================================

def display_product_card(
    rank,
    product_id,
    model_name,
    score=None,
    predicted_rating=None,
    hybrid_score=None,
    explanation=None
):

    product = get_product(
        product_id
    )

    if product is None:
        return

    title = product.get(
        "title"
    )

    if pd.isna(title):
        title = "Unknown Product"

    title = html.escape(
        str(title)
    )

    rating = product.get(
        "average_rating"
    )

    rating_number = product.get(
        "rating_number"
    )

    store = product.get(
        "store"
    )

    price = product.get(
        "price"
    )

    image_url = clean_image_url(
        product.get("image_url")
    )

    with st.container(
        border=True
    ):

        # Rank
        st.html(
            f'<div class="rank-badge">#{rank}</div>'
        )

        # Image
        if image_url:

            try:

                st.image(
                    image_url,
                    use_container_width=True
                )

            except Exception:

                st.caption(
                    "🖼️ Image unavailable"
                )

        else:

            st.caption(
                "🖼️ Image unavailable"
            )

        # Title
        st.html(
            f'<div class="product-title">{title}</div>'
        )

        # Rating
        if (
            not pd.isna(rating)
            or
            not pd.isna(rating_number)
        ):

            rating_text = (
                f"⭐ {float(rating):.1f}"
                if not pd.isna(rating)
                else ""
            )

            review_text = (
                f" · {int(rating_number):,} reviews"
                if not pd.isna(rating_number)
                else ""
            )

            st.html(
                f'<div class="product-meta">'
                f'{rating_text}{review_text}'
                f'</div>'
            )

        # Store
        if not pd.isna(store):

            st.html(
                f'<div class="product-meta">'
                f'🏪 {html.escape(str(store))}'
                f'</div>'
            )

        # Price
        if not pd.isna(price):

            try:

                price_text = (
                    f"${float(price):.2f}"
                )

            except (
                ValueError,
                TypeError
            ):

                price_text = html.escape(
                    str(price)
                )

            st.html(
                f'<div class="price">'
                f'{price_text}'
                f'</div>'
            )

        # Model score
        if (
            model_name == "SVD"
            and
            predicted_rating is not None
        ):

            st.html(
                f'<div class="match">'
                f'Predicted rating · '
                f'{predicted_rating:.2f}/5'
                f'</div>'
            )

        elif (
            model_name == "Hybrid"
            and
            hybrid_score is not None
        ):

            match_percent = int(
                round(
                    max(
                        0,
                        min(
                            1,
                            float(hybrid_score)
                        )
                    )
                    * 100
                )
            )

            st.html(
                f'<div class="match">'
                f'✨ {match_percent}% Match'
                f'</div>'
            )

        elif score is not None:

            st.html(
                f'<div class="match">'
                f'Recommendation score · '
                f'{score:.3f}'
                f'</div>'
            )

        # Explanation
        if explanation:

            st.html(
                f'<div class="explain">'
                f'{html.escape(explanation)}'
                f'</div>'
            )

        st.caption(
            f"ASIN: {product_id}"
        )


# ============================================================
# EXPLANATION
# ============================================================

def get_explanation(model_name):

    explanations = {

        "SVD":
            "Recommended from learned user-product "
            "preference patterns.",

        "KNN":
            "Recommended because it is close to products "
            "in the user's interaction history.",

        "Item-CF":
            "Recommended because it is similar to products "
            "the user previously interacted with.",

        "Hybrid":
            "Combines latent preferences, item similarity "
            "and product popularity."
    }

    return explanations.get(
        model_name,
        ""
    )


# ============================================================
# DISPLAY RECOMMENDATIONS
# ============================================================

def display_recommendations(
    recommendations,
    model_name
):

    headings = {

        "SVD":
            "Recommended for you",

        "KNN":
            "Similar products",

        "Item-CF":
            "Related products",

        "Hybrid":
            "AI-curated recommendations"
    }

    st.html(
        f'<div class="section-title">'
        f'{headings[model_name]}'
        f'</div>'
    )

    columns = st.columns(3)

    for rank, (_, row) in enumerate(
        recommendations.iterrows(),
        start=1
    ):

        with columns[
            (rank - 1) % 3
        ]:

            explanation = get_explanation(
                model_name
            )

            if model_name == "SVD":

                display_product_card(
                    rank,
                    row["parent_asin"],
                    model_name,
                    predicted_rating=row[
                        "predicted_rating"
                    ],
                    explanation=explanation
                )

            elif model_name == "Hybrid":

                display_product_card(
                    rank,
                    row["parent_asin"],
                    model_name,
                    hybrid_score=row[
                        "hybrid_score"
                    ],
                    explanation=explanation
                )

            else:

                display_product_card(
                    rank,
                    row["parent_asin"],
                    model_name,
                    score=row["score"],
                    explanation=explanation
                )


# ============================================================
# SECTION HELPER
# ============================================================

def section(
    title,
    subtitle=None
):

    st.html(
        f'<div class="section-title">'
        f'{html.escape(title)}'
        f'</div>'
    )

    if subtitle:

        st.html(
            f'<div class="section-subtitle">'
            f'{html.escape(subtitle)}'
            f'</div>'
        )


# ============================================================
# SIDEBAR
#
# IMPORTANT:
# No custom HTML is used for the sidebar brand.
# This prevents raw HTML from appearing.
# ============================================================

with st.sidebar:

    st.markdown(
        "# ✨ **RecomAI**"
    )

    st.caption(
        "AI-Powered Beauty Product "
        "Recommendation Engine"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Home",
            "Explore",
            "Analytics",
            "About"
        ]
    )


# ============================================================
# HOME PAGE
# ============================================================

if page == "Home":

    st.html("""
    <div class="hero">

        <h1>
            Find products you'll love.
        </h1>

        <p>
            RecomAI analyzes historical user-product interactions
            and combines collaborative filtering, latent-factor
            modeling and popularity signals to generate
            personalized beauty product recommendations.
        </p>

    </div>
    """)

    section(
        "Recommendation Studio",
        "Choose a user, recommendation model and result count."
    )

    control_col1, control_col2 = (
        st.columns(2)
    )

    # --------------------------------------------------------
    # USER
    # --------------------------------------------------------

    with control_col1:

        user_selection_mode = st.radio(
            "User",
            [
                "Demo User",
                "Manual User ID"
            ],
            horizontal=True
        )

        if (
            user_selection_mode
            ==
            "Demo User"
        ):

            demo_options = [
                (
                    f"Demo User {i} "
                    f"({user['interactions']} interactions)"
                )

                for i, user
                in enumerate(
                    demo_users,
                    start=1
                )
            ]

            selected_demo = st.selectbox(
                "Choose a demo user",
                demo_options
            )

            selected_index = (
                demo_options.index(
                    selected_demo
                )
            )

            user_id = demo_users[
                selected_index
            ]["user_id"]

            with st.expander(
                "Show selected User ID"
            ):

                st.code(
                    user_id
                )

        else:

            user_id = st.text_input(
                "Amazon User ID",
                placeholder=(
                    "Paste Amazon User ID"
                )
            )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    with control_col2:

        model_name = st.selectbox(
            "Recommendation model",
            [
                "SVD",
                "KNN",
                "Item-CF",
                "Hybrid"
            ]
        )

        top_n = st.slider(
            "Number of recommendations",
            min_value=1,
            max_value=10,
            value=6
        )

    icon, description = MODEL_INFO[
        model_name
    ]

    st.html(
        f"""
        <div class="model-box">

            <div class="model-title">
                {icon} {model_name}
            </div>

            <div class="model-description">
                {html.escape(description)}
            </div>

        </div>
        """
    )

    if model_name == "Hybrid":

        st.info(
            "Hybrid composition: "
            "**50% SVD + 30% Item-CF + 20% Popularity**"
        )

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    generate = st.button(
        "✨ Generate Recommendations",
        type="primary",
        use_container_width=True
    )

    if generate:

        if not user_id:

            st.warning(
                "Select a demo user or "
                "enter an Amazon User ID."
            )

        else:

            with st.spinner(
                f"Generating {model_name} recommendations..."
            ):

                if model_name == "SVD":

                    recommendations = (
                        recommend_svd(
                            user_id,
                            top_n
                        )
                    )

                elif model_name == "KNN":

                    recommendations = (
                        recommend_knn(
                            user_id,
                            top_n
                        )
                    )

                elif model_name == "Item-CF":

                    recommendations = (
                        recommend_item_cf(
                            user_id,
                            top_n
                        )
                    )

                else:

                    recommendations = (
                        recommend_hybrid(
                            user_id,
                            top_n
                        )
                    )

            st.session_state[
                "recommendations"
            ] = recommendations

            st.session_state[
                "recommendation_model"
            ] = model_name

            st.session_state[
                "recommendation_user"
            ] = user_id

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    if (
        "recommendations"
        in st.session_state
    ):

        recommendations = (
            st.session_state[
                "recommendations"
            ]
        )

        result_model = (
            st.session_state[
                "recommendation_model"
            ]
        )

        if recommendations.empty:

            st.error(
                "No recommendations could be "
                "generated for this user."
            )

        else:

            st.success(
                f"{len(recommendations)} "
                "recommendations generated."
            )

            display_recommendations(
                recommendations,
                result_model
            )

    # --------------------------------------------------------
    # METHODS
    # --------------------------------------------------------

    st.divider()

    section(
        "Recommendation methods"
    )

    method_columns = st.columns(4)

    for column, name in zip(
        method_columns,
        [
            "SVD",
            "KNN",
            "Item-CF",
            "Hybrid"
        ]
    ):

        icon, _ = MODEL_INFO[name]

        with column:

            st.html(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        {icon} MODEL
                    </div>

                    <div class="metric-value">
                        {name}
                    </div>

                </div>
                """
            )


# ============================================================
# EXPLORE PAGE
# ============================================================

elif page == "Explore":

    st.html("""
    <div class="hero">

        <h1>
            Explore the beauty catalog.
        </h1>

        <p>
            Search products by title or store,
            inspect product information and find
            similar products using Item-CF.
        </p>

    </div>
    """)

    search_col, rating_col = (
        st.columns([3, 1])
    )

    with search_col:

        search_query = st.text_input(
            "Search products",
            placeholder=(
                "e.g. moisturizer, shampoo, hair..."
            )
        )

    with rating_col:

        minimum_rating = st.slider(
            "Minimum rating",
            min_value=0.0,
            max_value=5.0,
            value=0.0,
            step=0.5
        )

    catalog = (
        product_catalog
        .dropna(subset=["title"])
        .copy()
    )

    if search_query:

        title_mask = (
            catalog["title"]
            .astype(str)
            .str.contains(
                search_query,
                case=False,
                na=False
            )
        )

        store_mask = (
            catalog["store"]
            .fillna("")
            .astype(str)
            .str.contains(
                search_query,
                case=False,
                na=False
            )
        )

        catalog = catalog[
            title_mask
            |
            store_mask
        ]

    if minimum_rating > 0:

        catalog = catalog[
            catalog[
                "average_rating"
            ].fillna(0)
            >= minimum_rating
        ]

    catalog = (
        catalog
        .sort_values(
            "rating_number",
            ascending=False
        )
        .head(30)
    )

    section(
        f"{len(catalog):,} products found"
    )

    if catalog.empty:

        st.info(
            "No products match your search."
        )

    else:

        explore_columns = (
            st.columns(3)
        )

        for rank, (_, product) in enumerate(
            catalog.iterrows(),
            start=1
        ):

            with explore_columns[
                (rank - 1) % 3
            ]:

                display_product_card(
                    rank=rank,
                    product_id=product[
                        "parent_asin"
                    ],
                    model_name="Explore"
                )

    st.divider()

    section(
        "Find similar products"
    )

    model_products = mappings[
        "products"
    ]

    def product_label(product_id):

        product = get_product(
            product_id
        )

        if product is None:
            return product_id

        title = product.get(
            "title"
        )

        if pd.isna(title):
            return product_id

        return str(title)[:90]

    selected_product = st.selectbox(
        "Select a product",
        model_products,
        format_func=product_label
    )

    similar_count = st.slider(
        "Number of similar products",
        1,
        10,
        5,
        key="similar_count"
    )

    if st.button(
        "🔗 Find Similar Products",
        use_container_width=True
    ):

        product_index = mappings[
            "product_to_index"
        ].get(selected_product)

        if product_index is None:

            st.error(
                "Product is not available "
                "in the model."
            )

        else:

            similarities = (
                item_similarity
                .getrow(product_index)
            )

            candidates = []

            for index, similarity in zip(
                similarities.indices,
                similarities.data
            ):

                if index == product_index:
                    continue

                if similarity <= 0:
                    continue

                candidates.append(
                    (
                        mappings[
                            "products"
                        ][index],

                        float(similarity)
                    )
                )

            candidates = sorted(
                candidates,
                key=lambda x: x[1],
                reverse=True
            )[:similar_count]

            if not candidates:

                st.info(
                    "No similar products were found."
                )

            else:

                similar_columns = (
                    st.columns(3)
                )

                for rank, (
                    product_id,
                    similarity
                ) in enumerate(
                    candidates,
                    start=1
                ):

                    with similar_columns[
                        (rank - 1) % 3
                    ]:

                        display_product_card(
                            rank=rank,
                            product_id=product_id,
                            model_name="Item-CF",
                            score=similarity,
                            explanation=(
                                "Similar to the selected "
                                "product based on item-level "
                                "cosine similarity."
                            )
                        )


# ============================================================
# ANALYTICS PAGE
# ============================================================

elif page == "Analytics":

    st.html("""
    <div class="hero">

        <h1>
            Model Analytics
        </h1>

        <p>
            A compact view of the recommendation system,
            dataset scale and evaluation results.
        </p>

    </div>
    """)

    metrics = [
        (
            "Users",
            "44,155"
        ),
        (
            "Products",
            "20,987"
        ),
        (
            "Interactions",
            "88,895"
        ),
        (
            "Train / Test",
            "55,496 / 27,176"
        )
    ]

    metric_columns = st.columns(4)

    for column, (
        label,
        value
    ) in zip(
        metric_columns,
        metrics
    ):

        with column:

            st.html(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        {label}
                    </div>

                    <div class="metric-value">
                        {value}
                    </div>

                </div>
                """
            )

    section(
        "Recommendation performance"
    )

    st.caption(
        "Top-5 ranking evaluation from "
        "the project experiments."
    )

    comparison = pd.DataFrame(
        {
            "Model": [
                "Popularity",
                "Item-CF",
                "KNN",
                "SVD"
            ],

            "Precision@5": [
                0.00000,
                0.00365,
                0.00476,
                0.00000
            ],

            "Recall@5": [
                0.00000,
                0.01826,
                0.02381,
                0.00000
            ],

            "NDCG@5": [
                0.00000,
                0.01302,
                0.00921,
                0.00000
            ]
        }
    )

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    section(
        "SVD rating prediction"
    )

    rating_metrics = pd.DataFrame(
        {
            "Metric": [
                "MAE",
                "RMSE"
            ],

            "SVD": [
                1.0622,
                1.3358
            ],

            "Global Mean Baseline": [
                1.1182,
                1.3818
            ]
        }
    )

    st.dataframe(
        rating_metrics,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Top-N ranking and rating prediction measure "
        "different objectives. SVD should not be judged "
        "only by the Top-5 exact-match results."
    )

    section(
        "System architecture"
    )

    architecture = [

        (
            "01",
            "User History",
            "Historical ratings and product interactions."
        ),

        (
            "02",
            "Model Layer",
            "SVD, KNN and Item-CF generate candidate products."
        ),

        (
            "03",
            "Hybrid Layer",
            "Combines model signals and popularity."
        ),

        (
            "04",
            "Product Layer",
            "Metadata and images turn predictions into a usable UI."
        )
    ]

    architecture_columns = (
        st.columns(4)
    )

    for column, item in zip(
        architecture_columns,
        architecture
    ):

        number, title, description = item

        with column:

            st.html(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        {number}
                    </div>

                    <div class="metric-value">
                        {title}
                    </div>

                    <p style="
                        color:#A1A1B5;
                        font-size:13px;
                        line-height:1.5;
                    ">
                        {description}
                    </p>

                </div>
                """
            )


# ============================================================
# ABOUT PAGE
# ============================================================

else:

    st.html("""
    <div class="hero">

        <h1>
            About RecomAI
        </h1>

        <p>
            RecomAI is an AI-powered beauty product
            recommendation engine built around
            collaborative filtering and machine
            learning techniques.
        </p>

    </div>
    """)

    about_col1, about_col2 = (
        st.columns(2)
    )

    with about_col1:

        section(
            "Project objective"
        )

        st.write(
            """
            The objective is to transform historical
            Amazon Beauty product interactions into
            personalized product recommendations.

            The system compares multiple recommendation
            strategies and exposes the results through
            an interactive Streamlit application.
            """
        )

    with about_col2:

        section(
            "Machine learning"
        )

        st.write(
            """
            RecomAI uses:

            • Singular Value Decomposition (SVD)

            • Item-based Collaborative Filtering

            • K-Nearest Neighbors (KNN)

            • Popularity-based ranking

            • Hybrid recommendation
            """
        )

    st.divider()

    section(
        "Technology stack"
    )

    stack = [

        (
            "Python",
            "Core programming"
        ),

        (
            "Pandas",
            "Data processing"
        ),

        (
            "Scikit-learn",
            "Similarity and ML"
        ),

        (
            "Streamlit",
            "Interactive UI"
        )
    ]

    stack_columns = st.columns(4)

    for column, (
        name,
        description
    ) in zip(
        stack_columns,
        stack
    ):

        with column:

            st.html(
                f"""
                <div class="metric-card">

                    <div class="metric-value">
                        {name}
                    </div>

                    <div class="metric-label">
                        {description}
                    </div>

                </div>
                """
            )

    section(
        "Recommendation models"
    )

    for model_name, (
        icon,
        description
    ) in MODEL_INFO.items():

        with st.expander(
            f"{icon} {model_name}"
        ):

            st.write(
                description
            )

            if model_name == "Hybrid":

                st.write(
                    "Weights: 50% SVD + "
                    "30% Item-CF + "
                    "20% Popularity."
                )

    st.divider()

    section(
        "Project identity"
    )

    st.html("""
    <p style="
        color:#A1A1B5;
        line-height:1.7;
    ">
        RecomAI<br>
        AI-Powered Beauty Product Recommendation Engine<br>
        Built with Python, Pandas, Scikit-learn,
        Surprise, SciPy and Streamlit.
    </p>
    """)


# ============================================================
# FOOTER
# ============================================================

st.html("""
<div class="footer">

    RecomAI • AI-Powered Beauty Product
    Recommendation Engine

    <br>

    Built with Python • Pandas • Scikit-learn •
    Surprise • SciPy • Streamlit

</div>
""")