# Gets a dataset of movie reviews labeled as either positive or negative
import nltk
nltk.download('movie_reviews')

from nltk.corpus import movie_reviews
import random

# Load the movie reviews data and format it correctly
# The format is as follows:
# "The movie was great" -> (['the', 'movie', 'was', 'great'], pos)
# It splits up the review into individual words and labels it as positive or negative
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Shuffle the reviews to have a random dataset of positive and negative reviews
random.shuffle(documents)

from nltk.corpus import stopwords
from nltk import FreqDist
import string

# This removes common filler words like "the", "is", "on", etc that don't contribute to it being positive or negative
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Build a frequency distribution of all words (basically the words that appear more often will be higher up on this list)
all_words = [w.lower() for w in movie_reviews.words()
             if w.lower() not in stop_words and w not in string.punctuation]
all_words_freq = FreqDist(all_words)

# Take top 4000 words as indicators for if its positive or negative
word_features = list(all_words_freq)[:4000]

# Convert documents to tokens for the model to process
# This basically just goes through the movie review and detects which of the top 4000 words appear in the review
def document_features(doc_words):
    words = set(doc_words)
    features = {}
    for w in word_features:
        features[f'contains({w})'] = (w in words)
    return features

# This labels the review as either positive or negative based on the 4000 common words it has and hwo they correlate to positive or negative
featuresets = [(document_features(d), c) for (d, c) in documents]

# All this is to train the model to be more accurate; It uses the imported movie review database to train
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

# Splits data into training and testing sets (80% training, 20% testing)
X = [f for (f, c) in featuresets]
y = [c for (f, c) in featuresets]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Turns the review lists from earlier into numbers and tries to recognize patterns
# Vectorize is just fancy for turn into number and process
# MultinomialNB detects patterns that appear in the numbers
vec = DictVectorizer()
clf = MultinomialNB()
pipeline = make_pipeline(vec, clf)
pipeline.fit(X_train, y_train)

# Predicts the tone for the tester reviews
# Also prints out all the diagnostic stuff like precision, confidence score, etc
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Function to predict the sentiment (positive or negative) of testing reviews
def predict_sentiment(text):
    words = text.lower().split()
    words = [w for w in words if w not in stop_words and w not in string.punctuation]
    feats = document_features(words)
    return pipeline.predict([feats])[0]

# Testing
print(predict_sentiment("I really loved the acting and the story."))
print(predict_sentiment("This movie was boring and way too long."))
print(predict_sentiment("The plot is amazing and I loved the cinematography."))
print(predict_sentiment("The plot was awful and the entire movie was shallow."))
print(predict_sentiment("Bad acting. Bad CGI. Terrible movie."))
print(predict_sentiment("The camerawork is stunning and the transitions are really good."))
print(predict_sentiment("I did not find it boring in any way."))