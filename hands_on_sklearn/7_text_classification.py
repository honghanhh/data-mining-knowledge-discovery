
# Large Movie Review Dataset

# First download the data from the following URL an unzip it.
# using http://ai.stanford.edu/~amaas/data/sentiment/
# Change four paths to the data in lines 35 - 43:
# positive_reviews_dir, negative_reviews_dir, test_positive_reviews_dir, test_negative_reviews_dir


# When you first use the stopwords and the wordnet lemmatizer, there will be an error (resource not found).
# The following two lines of code will download the resources.

# nltk.download('stopwords')
# nltk.download('wordnet')


# ---------------------------------------------------------------------------------------
print("Loading the train and test corpus")
import os
import pandas as pd


def load_data(directory, label, instances=100000):  # one document per file
    list_of_docs = []
    list_of_labels = []
    for file in os.listdir(directory):
        if instances and len(list_of_docs) > instances:
            break
        fname = os.path.join(directory, file)
        with open(fname, encoding="utf8") as text_file:
            text = text_file.read()
            list_of_docs.append(text)
            list_of_labels.append(label)
    df = pd.DataFrame()
    df['text'] = list_of_docs
    df['label'] = list_of_labels
    return df


positive_reviews_dir = r"d:/Data/aclImdb/train/pos"
negative_reviews_dir = r"d:/Data/aclImdb/train/neg"
train_corpus = load_data(positive_reviews_dir, "positive")                                                              #train_corpus = pd.DataFrame(columns=['text', 'label'])
train_corpus = train_corpus.append(load_data(negative_reviews_dir, "negative"), ignore_index=True)
print("Train corpus sample:\n", train_corpus.head())
print("Train corpus shape:", train_corpus.shape)

test_positive_reviews_dir = r"d:/Data/aclImdb/test/pos"
test_negative_reviews_dir = r"d:/Data/aclImdb/test/neg"
test_corpus = load_data(test_positive_reviews_dir, "positive")                                                          #test_corpus = pd.DataFrame(columns=['text', 'label'])
test_corpus = test_corpus.append(load_data(test_negative_reviews_dir, "negative"), ignore_index=True)
print("Test corpus shape:", test_corpus.shape)


# ---------------------------------------------------------------------------------------

print("Initialize text pre-processing")
import nltk

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
stopwords = nltk.corpus.stopwords.words("english")
lem = nltk.stem.wordnet.WordNetLemmatizer()


def preprocess(text):
    tokens = tokenizer.tokenize(text)                                                   # tokenizacija s pomočjo privzete NLTK funkcije za tokenizacijo
    lowercased = [word.lower() for word in tokens]                                      # male črke
    without_stopwords = [word for word in lowercased if word not in stopwords]          # odstranitev blokiranih besed
    lemmatized_tokens = [lem.lemmatize(x) for x in without_stopwords]                   # lematizacija
    return lemmatized_tokens

# ---------------------------------------------------------------------------------------
print("Tf-idf transform")
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(tokenizer=preprocess, stop_words=None, max_features=5000)

tfidf.fit(train_corpus['text'])
train_data = tfidf.transform(train_corpus['text'])
test_data = tfidf.transform(test_corpus['text'])

#feature_names = tfidf.get_feature_names()

# -------------------------------------------------------------------------------------------
print("Train and test classification models")

from sklearn import metrics
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import svm

classifiers = [
    ("Naive Bayes", naive_bayes.MultinomialNB()),
    ("Logistic regression", linear_model.LogisticRegression(solver='lbfgs')),
    ("SCV", svm.LinearSVC())]

for name, classifier in classifiers:
    classifier.fit(train_data, train_corpus['label'])
    predictions = classifier.predict(test_data)
    print(name, metrics.accuracy_score(predictions, test_corpus['label']))

# -------------------------------------------------------------------------------------------
print("Predict on new data")
new_documents = ["I loved this movie.",
                "The storyline was boring, while the action was great.",
                "The end was very unexpected, I didn't let me sleep for a week."]

new_data = tfidf.transform(new_documents)

for name, classifier in classifiers:
    predictions = classifier.predict(new_data)
    print(predictions)


