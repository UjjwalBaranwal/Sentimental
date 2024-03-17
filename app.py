import spacy
import string
import pickle
import streamlit as st
from spacy.lang.en.stop_words import STOP_WORDS

# Load English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe("sentencizer", before="parser")
stopwords = list(STOP_WORDS)
punct = string.punctuation

# Define the text_data_cleaning function


def text_data_cleaning(sentence):
    doc = nlp(sentence)
    tokens = []
    for token in doc:
        if token.lemma_ != "-PRON-":
            temp = token.lemma_.lower().strip()
        else:
            temp = token.lower()
        tokens.append(temp)

    cleaned_tokens = []
    for token in tokens:
        if token not in stopwords and token not in punct:
            cleaned_tokens.append(token)
    cleaned_text = " ".join(cleaned_tokens)
    return cleaned_text


# Load the sentiment analysis model and TF-IDF vectorizer
with open('clf.pkl', 'rb') as f:
    clf = pickle.load(f)
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Define the predict_sentiment function


def predict_sentiment(sentence):
    cleaned_text = text_data_cleaning(sentence)
    comment_vector = tfidf.transform([cleaned_text])
    sentiment = clf.predict(comment_vector)[0]
    return sentiment


def main():
    st.header("Sentimental")
    # query = st.text_input("Write your Sentiment")
    # if query:
    #     if st.button('Predict Sentiment'):
    #         sentiment = predict_sentiment(query)
    #         st.write('Predicted Sentiment:', sentiment)
    st.write(predict_sentiment("i am a good boy"))


if __name__ == '__main__':
    main()
