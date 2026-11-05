
from transformers import pipeline
import re

sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

def analyze_sentiment(review):
    full_result = sentiment_analyzer(review)[0]

    contrast_pattern = r"\b(but|however|although|though|yet)\b"
    parts = re.split(contrast_pattern, review, maxsplit=1, flags=re.IGNORECASE)

    if len(parts) == 3:
        first_part = parts[0].strip()
        second_part = parts[2].strip()

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


reviews = [
    "This movie was terrible boring awful and disappointing",
    "This movie was wonderful amazing fantastic and enjoyable",
    "The movie was good but the ending was terrible",
    "This movie was absolutely terrible and boring",
    "This movie was absolutely great",
    "This movie is great",
    "The acting was amazing but the story was boring"
]

for review in reviews:
    result = analyze_sentiment(review)

    print("\nReview:", review)
    print("Sentiment:", result["sentiment"])
    print("Score:", round(result["score"], 4))

    if result["details"]:
        print("Sentiment breakdown:")
        for text, sentiment in result["details"]:
            print(f"  {text} -> {sentiment}")
