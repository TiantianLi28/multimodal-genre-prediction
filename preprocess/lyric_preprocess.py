import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_lyrics(lyrics):
    """
    for preprocessing the lyrics to remove stop words, lemmatize, etc
    :param lyrics: str, lyrics
    :return: preprocessed string of lyrics
    """
    lyrics = lyrics.split()
    lyrics_list = []
    lyric = ""
    # split by capitalized words (not perfect)
    for i in range(len(lyrics)):
        word = lyrics[i]
        a = word.istitle()
        n = len(lyrics_list)
        if (word.istitle()):
            if lyric == "":
                lyric = word
            else:
                lyrics_list.append(lyric)
                lyric = word
        else:
            lyric += " " + word
            if i + 1 == len(lyrics):
                lyrics_list.append(lyric)
    processed_lyrics_list = []
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
    model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = model.encode(lyrics)
    for sentence, embedding in zip(lyrics, embeddings):
        print("Sentence:", sentence)
        print("Embedding:", embedding)


if __name__ == "__main__":
    df = pd.read_csv('dat/filtered_data.csv')
    test = df['lyrics'][0]
    lyrics = preprocess_lyrics(test)
    # print(lyrics)
    transform(lyrics)