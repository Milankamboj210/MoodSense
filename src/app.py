
import streamlit as st
import re
from transformers import pipeline

st.set_page_config(
    page_title="MoodSense",
    page_icon="🎬",
    layout="centered"
)

@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )

sentiment_analyzer = load_model()

def analyze_sentiment(review):
    full_result = sentiment_analyzer(review)[0]

    contrast_pattern = r"\b(but|however|although|though|yet)\b"
    parts = re.split(
        contrast_pattern,
        review,
        maxsplit=1,
        flags=re.IGNORECASE
    )

    if len(parts) == 3:
        first_part = parts[0].strip()
        second_part = parts[2].strip()

        if first_part and second_part:
            first_result = sentiment_analyzer(first_part)[0]
            second_result = sentiment_analyzer(second_part)[0]

            if first_result["label"] != second_result["label"]:
                return {
                    "sentiment": "MIXED",
                    "score": full_result["score"],
                    "details": [
                        (first_part, first_result["label"]),
                        (second_part, second_result["label"])
                    ]
                }

    return {
        "sentiment": full_result["label"],
        "score": full_result["score"],
        "details": []
    }

st.title("🎬 MoodSense")
st.subheader("AI-Powered Movie Review Sentiment Analyzer")

st.write(
    "Discover the emotion behind movie reviews using "
    "Natural Language Processing and Deep Learning."
)

review = st.text_area(
    "Enter your movie review:",
    placeholder="Example: The acting was amazing but the story was boring...",
    height=150
)

if st.button("Analyze Sentiment", type="primary", use_container_width=True):
    if not review.strip():
        st.warning("Please enter a movie review first.")
    else:
        with st.spinner("Analyzing your review..."):
            try:
                result = analyze_sentiment(review.strip())

                sentiment = result["sentiment"]

                st.divider()
                st.subheader("Analysis Result")

                if sentiment == "POSITIVE":
                    st.success("😊 POSITIVE SENTIMENT")
                elif sentiment == "NEGATIVE":
                    st.error("😞 NEGATIVE SENTIMENT")
                else:
                    st.warning("🤔 MIXED SENTIMENT")

                st.metric(
                    "Model confidence for full-review prediction",
                    f"{result['score'] * 100:.2f}%"
                )

                if result["details"]:
                    st.subheader("Sentiment Breakdown")

                    for text, label in result["details"]:
                        if label == "POSITIVE":
                            st.success(f"{text} → Positive")
                        else:
                            st.error(f"{text} → Negative")

                st.caption(
                    "Powered by DistilBERT. Mixed sentiment is detected "
                    "using contrast-based clause analysis."
                )

            except Exception as error:
                st.error(f"Something went wrong: {error}")
