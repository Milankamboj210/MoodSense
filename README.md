# MoodSense 🎬
### AI-Powered Movie Review Sentiment Analyzer

MoodSense is a movie review sentiment analysis application that uses Natural Language Processing (NLP) and Deep Learning to classify reviews as **Positive**, **Negative**, or **Mixed**.

## Features

- Movie review sentiment analysis.
- Positive and negative sentiment classification.
- Mixed sentiment detection using contrast-based clause analysis.
- Sentiment breakdown for contrasting review statements.
- Interactive web interface built with Streamlit.
- Pretrained DistilBERT model for sentiment inference.
- Custom-trained SimpleRNN model using the IMDB movie review dataset.

## Technologies Used

- Python
- TensorFlow / Keras
- SimpleRNN
- DistilBERT
- Hugging Face Transformers
- PyTorch
- Streamlit
- IMDB Dataset

## Project Structure

```text
MoodSense/
│
├── data/
├── models/
│   ├── moodsense_model.keras
│   └── moodsense_model_v2.keras
│
├── src/
│   ├── inspect_data.py
│   ├── prepare_data.py
│   ├── embedding_demo.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── predict.py
│   ├── predict_pretrained.py
│   └── app.py
│
├── requirements.txt
└── README.md
```

*Adjust the model filenames or folder contents if your local project differs.*

## Models

### 1. SimpleRNN

A recurrent neural network trained on the IMDB movie review dataset.

- Vocabulary size: 10,000
- Maximum review length: 500 tokens
- Embedding dimension: 128
- SimpleRNN units: 128
- Output layer: Sigmoid activation
- Loss function: Binary cross-entropy
- Optimizer: Adam

The stabilized V2 model achieved approximately **77.70% test accuracy**.

### 2. DistilBERT

MoodSense also integrates the pretrained `distilbert-base-uncased-finetuned-sst-2-english` model from Hugging Face for sentiment prediction.

### 3. Mixed Sentiment Detection

A contrast-based heuristic identifies conflicting sentiments in clauses separated by expressions such as "but", "however", and "although".

This is an additional rule-based layer, not a third class learned by the pretrained DistilBERT model.

## Installation and Setup

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd MoodSense
```

Create and activate the Conda environment if needed:

```bash
conda create -n moodsense python=3.11
conda activate moodsense
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run src/app.py
```

The application will open in your browser.

## Future Improvements

- Train a dedicated three-class sentiment model.
- Improve mixed sentiment detection using aspect-based sentiment analysis.
- Add review history and sentiment visualizations.
- Deploy the application online.

## Author

**Milan Kamboj**

Developed as a hands-on Deep Learning and Natural Language Processing project.