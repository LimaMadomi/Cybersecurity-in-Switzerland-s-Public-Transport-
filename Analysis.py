# analysis.py
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

# Load the preprocessed dataset
preprocessed_file_path = 'preprocessed_data.csv'
data = pd.read_csv(preprocessed_file_path)

# Function to perform sentiment analysis
def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Perform sentiment analysis on the headlines
data['Headline_Sentiment'] = data['Cleaned_Headline'].apply(analyze_sentiment)

# Extract top 20 keywords using TF-IDF
vectorizer = TfidfVectorizer(max_features=20)
tfidf_matrix = vectorizer.fit_transform(data['Cleaned_Headline'])
keywords = vectorizer.get_feature_names_out()

# Calculate reporting counts
data['Date'] = pd.to_datetime(data['Data']).dt.date
reporting_counts = data['Date'].value_counts().sort_index()

# Save analysis results
analysis_results_file = 'analysis_results.csv'
data.to_csv(analysis_results_file, index=False)
print(f"Analysis results saved to {analysis_results_file}")

# Save top 20 keywords and reporting counts
keywords_file = 'top_20_keywords.txt'
with open(keywords_file, 'w') as f:
    for keyword in keywords:
        f.write(f"{keyword}\n")
print(f"Top 20 keywords saved to {keywords_file}")

reporting_counts_file = 'reporting_counts.csv'
reporting_counts.to_csv(reporting_counts_file, index=True)
print(f"Reporting counts saved to {reporting_counts_file}")
