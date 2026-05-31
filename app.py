import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Movie Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = tf.keras.models.load_model(
    "models/rnn_model.keras"
)

word_index = imdb.get_word_index()

MAX_LEN = 200

# --------------------------------------------------
# PREPROCESS FUNCTION
# --------------------------------------------------

def preprocess_text(text):

    words = text.lower().split()

    sequence = []

    for word in words:

        if word in word_index:

            sequence.append(
                word_index[word] + 3
            )

    padded = pad_sequences(
        [sequence],
        maxlen=MAX_LEN
    )

    return padded


# --------------------------------------------------
# CSS
# --------------------------------------------------

st.markdown("""
<style>

.stApp{
    background: linear-gradient(
        135deg,
        #fdf4ff,
        #f3e8ff,
        #ede9fe
    );
}

/* Title */
.title{
    text-align:center;
    color:#6d28d9;
    font-size:55px;
    font-weight:bold;
}

/* Cards */
.card{
    background:white;
    padding:20px;
    border-radius:20px;
    text-align:center;
    box-shadow:0px 4px 12px rgba(0,0,0,0.08);
}

/* Text Colors */
.card h2{
    color:#6d28d9;
}

.card h3{
    color:#4c1d95;
}

.section{
    color:#4c1d95;
    font-weight:bold;
}

/* Metrics */
[data-testid="stMetricValue"]{
    color:#6d28d9 !important;
    font-weight:bold;
}

[data-testid="stMetricLabel"]{
    color:#4c1d95 !important;
    font-weight:bold;
}

/* Labels */
label{
    color:#4c1d95 !important;
    font-weight:bold;
}

/* Text Area */
textarea{
    background:white !important;
    color:black !important;
}

/* Button */
.stButton > button{
    width:100%;
    height:55px;
    border-radius:12px;
    font-size:18px;
    font-weight:bold;
}

            .section{
    color:#4c1d95 !important;
    font-weight:bold;
}

/* Text Area Label */
label{
    color:#4c1d95 !important;
    font-weight:bold;
}

/* Text entered by user */
textarea{
    color:#000000 !important;
    background:white !important;
}

/* Placeholder text */
textarea::placeholder{
    color:#666666 !important;
    opacity:1 !important;
}

/* Markdown headings */
h1,h2,h3{
    color:#4c1d95 !important;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.markdown(
"""
<div class='title'>
🎬 Movie Review Sentiment Analysis Using RNN
</div>
""",
unsafe_allow_html=True
)

st.write("")

# --------------------------------------------------
# DATASET OVERVIEW
# --------------------------------------------------

st.markdown(
"""
<h2 class='section'>
📊 Dataset Overview
</h2>
""",
unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class='card'>
        <h3>Training Samples</h3>
        <h2>25,000</h2>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class='card'>
        <h3>Testing Samples</h3>
        <h2>25,000</h2>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class='card'>
        <h3>Vocabulary Size</h3>
        <h2>10,000</h2>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class='card'>
        <h3>Classes</h3>
        <h2>2</h2>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# --------------------------------------------------
# REVIEW INPUT
# --------------------------------------------------

st.markdown(
"""
<h2 style="
color:#4c1d95;
font-weight:bold;
font-size:38px;
">
✍ Enter Movie Review
</h2>
""",
unsafe_allow_html=True
)

review = st.text_area(
    "",
    height=220,
    placeholder="Type your movie review here..."
)

predict = st.button(
    "Predict Sentiment"
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if predict:

    if review.strip() == "":

        st.warning(
            "Please enter a movie review."
        )

    else:

        review_seq = preprocess_text(
            review
        )

        prediction = model.predict(
            review_seq,
            verbose=0
        )

        score = float(
            prediction[0][0]
        )

        if score >= 0.5:
            st.success("😊 Positive Review")
            st.metric(
                    "Confidence Score",
                        f"{score*100:.2f}%"
            )
            st.info(
        "This review expresses positive sentiment."
    )
        else:
            st.error("😞 Negative Review")
            st.metric(
        "Confidence Score",
        f"{(1-score)*100:.2f}%"
    )
            st.info(
        "This review expresses negative sentiment."
    )