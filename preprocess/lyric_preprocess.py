import pandas as pd
import numpy as np
import nltk, spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def divide_lyrics(lyrics):
    """ dividing lyrics"""
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(lyrics)
    verses = []
    current_verse = []

    for token in doc:
        if token.is_sent_start:
            if current_verse:
                verses.append(" ".join(current_verse))
                current_verse = []
        current_verse.append(token.text)

    if current_verse:
        verses.append(" ".join(current_verse))
    return verses


def preprocess_lyrics(lyrics):
    """
    for preprocessing the lyrics to remove stop words, lemmatize, etc
    :param lyrics: str, lyrics
    :return: preprocessed string of lyrics
    """
    processed_lyrics_list = []
    lyrics_list = divide_lyrics(lyrics)
    for lyric in lyrics_list:
        words = word_tokenize(lyric)

        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word.lower() not in stop_words]

        # lemmatize
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
        processed_lyrics = ' '.join(lemmatized_words)
        processed_lyrics_list.append(processed_lyrics)
    return processed_lyrics_list

def transform(lyrics):
    """
    using sentence transformer to initiate word embeddings on lyrics
    :param lyrics: str, preprocessed lyrics
    :return:
    """
    lyrics = preprocess_lyrics(lyrics)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = model.encode(lyrics)
    average_embeddings = np.mean(embeddings, axis=1)
    # print(average_embeddings)
    return average_embeddings

if __name__ == "__main__":
    df = pd.read_csv('dat/data_with_genre713.csv')
    lyrics = df['lyrics']
    embeddings = []
    for lyric in tqdm(lyrics):
        embedding = transform(lyric)
        embeddings.append(embedding)
    df['lyric-embeddings'] = embeddings
    df.to_csv('dataset_with_embeddings_2120.csv', index=False)
    # print(lyrics)