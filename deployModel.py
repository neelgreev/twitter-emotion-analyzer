import pandas as pd

df = pd.read_csv('Dataset - Test.csv')

df.head()

import re

brands = [
    "apple", "samsung", "tesla", "microsoft", "google",
    "amazon", "zomato", "swiggy", "flipkart", "nike",
    "adidas", "coca cola", "pepsi", "starbucks"
]

def detect_brand(text):
    text = str(text).lower()
    for brand in brands:
        # match whole word to avoid false positives (e.g., "apple" in "pineapple")
        if re.search(rf"\b{brand}\b", text):
            return brand.capitalize()
    return None

df['Brand'] = df['Tweet'].apply(detect_brand)
df = df[df['Brand'].notnull()]  # keep only brand-related tweets


import pickle
with open('twitter-model.pkl','rb') as f:
  pipeline = pickle.load(f)


df['PredictedEmotion']= pipeline.predict(df['Tweet'])


import streamlit as st
# ===== Streamlit App =====
st.set_page_config(page_title="Brand Reputation Monitor", layout="wide")
st.title("📊 Brand Reputation Monitor")

# Brand Selector
brand_options = sorted(df['Brand'].unique())
selected_brands = st.multiselect("Select brands to analyze", brand_options, default=brand_options[:2])
filtered_df = df[df['Brand'].isin(selected_brands)]

# ===== Emotion Distribution =====
st.subheader("Emotion Distribution by Brand")
if not filtered_df.empty:
    emotion_counts = filtered_df.groupby('Brand')['PredictedEmotion'].value_counts().unstack().fillna(0)
    st.bar_chart(emotion_counts)
else:
    st.write("No tweets found for selected brands.")


st.write("0: SAD , 1: JOY, 2: LOVE , 3: ANGER, 4: FEAR")

# ===== Sentiment Breakdown (if dataset has Sentiment column) =====
if 'Sentiment' in filtered_df.columns:
    st.subheader("Sentiment Breakdown by Brand")
    sentiment_counts = filtered_df.groupby('Brand')['Sentiment'].value_counts().unstack().fillna(0)
    st.bar_chart(sentiment_counts)



# ===== Live Tweets Feed =====
st.subheader("Live Tweets Feed (Random Sample)")
if not filtered_df.empty:
    sample_tweets = filtered_df.sample(min(10, len(filtered_df)))
    for _, row in sample_tweets.iterrows():
        st.write(f"**[{row['Brand']}]** {row['Tweet']} — *Emotion: {row['PredictedEmotion']}*")