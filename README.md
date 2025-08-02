Brand Reputation Monitor
An interactive Streamlit dashboard for monitoring brand reputation using Twitter-like text data.
The app detects brands mentioned in tweets, classifies their emotions using a pre-trained ML pipeline, and visualizes insights with charts, word clouds, and a live-like tweet feed.

Features
Brand Detection: Automatically identifies brands mentioned in tweets.

Emotion Classification: Uses a pre-trained TF‑IDF + ML model to detect emotions (e.g., joy, anger, sadness).

Sentiment Analysis: (If available in the dataset).

Visualizations:

Emotion distribution by brand.

Sentiment breakdown (if dataset has sentiment labels).

Word clouds for trending keywords.

Live-like Feed: Displays random brand-related tweets with predicted emotions.

Tech Stack:
Python: Core programming language.

Streamlit: For building the interactive dashboard.

Pandas: Data manipulation.

scikit-learn: Pre-trained TF‑IDF + ML pipeline for emotion detection.

