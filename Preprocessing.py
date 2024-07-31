# data_preprocessing.py
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords

# Ensure NLTK data is downloaded
nltk.download('stopwords', quiet=True)

# Define a function to preprocess the text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# Load the dataset
file_path = 'News.xlsx'  # Replace with the actual path to your file
data = pd.read_excel(file_path)

# Preprocess headlines
data['Cleaned_Headline'] = data['Headline'].apply(preprocess_text)

# Save the preprocessed data
preprocessed_file_path = 'preprocessed_data.csv'
data.to_csv(preprocessed_file_path, index=False)
print(f"Preprocessed data saved to {preprocessed_file_path}")
