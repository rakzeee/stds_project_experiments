import pandas as pd
import numpy as np
import nltk
#nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus.reader import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import time
import logging

# Set random seed
np.random.seed(2)

# Read dataset
def read_and_split():
    # I am using amazon review dataset available here http://jmcauley.ucsd.edu/data/amazon/
    dataset = pd.read_csv("review_dataset.csv", encoding='latin-1')

    # Text classification algorithms are often used along with data preprocessing
    # It includes removing blank rows, lower casing, word tokenization, removing stop words, removing non-alpha text, lemamtization

    dataset['text'].dropna(inplace=True)

    dataset['text'] = [entry.lower() for entry in dataset['text']]

    dataset['text'] = [word_tokenize(entry) for entry in dataset['text']]

    # apply tagging using wordnet
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV

    for index, entry in enumerate(dataset['text']):
        result_words = []

        lemmatizer = WordNetLemmatizer()
        for word, tag in pos_tag(entry):
            if word not in stopwords.words('english') and word.isalpha():
                res_word = lemmatizer.lemmatize(word, tag_map[tag[0]])
                result_words.append(res_word)
        dataset.loc[index, 'text_final'] = str(result_words)

    print(dataset['text_final'].head())

    # split to train/test data
    train_x, test_x, train_y, test_y = model_selection.train_test_split(dataset['text_final'], dataset['label'], test_size=0.3)
    return dataset, train_x, test_x, train_y, test_y


def extract_features(dataset, train_x, test_x):
    # vectorize the words using TF-IDF
    vectors = TfidfVectorizer(max_features=5000)
    vectors.fit(dataset['text_final'])

    tfidf_train_x = vectors.transform(train_x)
    tfidf_test_x = vectors.transform(test_x)
    return tfidf_train_x, tfidf_test_x


dataset, train_x, test_x, train_y, test_y = read_and_split()
# transform string label values to numerical labels
Encoder = LabelEncoder()
train_y = Encoder.fit_transform(train_y)
test_y = Encoder.fit_transform(test_y)
tfidf_train_x, tfidf_test_x = extract_features(dataset, train_x, test_x)

# t = time.time()
# # Classifier algorithm - Naive Bayes
# # fit the training dataset on the classifier
# naive = naive_bayes.MultinomialNB()
# naive.fit(tfidf_train_x, train_y)
# time_taken = time.time() - t
# print(f'Time taken to train Naive Bayes classifier: {time_taken} s.')
#
# t = time.time()
# # predict the labels on test_x
# predictions_NB = naive.predict(tfidf_test_x)
# time_taken = time.time() - t
# print(f'Time taken to predict all values from test set (Naive Bayes): {time_taken} s.')
#
# # show accuracy
# print("Naive Bayes Accuracy Score -> ", accuracy_score(predictions_NB, test_y) * 100)
#
#
# # Classifier algorithm - SVM
# t = time.time()
# SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
# SVM.fit(tfidf_train_x, train_y)

# print(f'Time taken to train SVM classifier: {time_taken} s.')
#
# t = time.time()
# predictions_SVM = SVM.predict(tfidf_test_x)
# time_taken = time.time() - t
# print(f'Time taken to predict all values from test set (SVM): {time_taken} s.')
#
# print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, test_y) * 100)

t = time.time()
logging.info("Training a Logistic Regression Model...")
scikit_log_reg = LogisticRegression(verbose=1, solver='liblinear',random_state=0, C=5, penalty='l2',max_iter=1000)
lr_model = scikit_log_reg.fit(tfidf_train_x,train_y)
time_taken = time.time() - t
print(f'Time taken to train LR classifier: {time_taken} s.')
t = time.time()
predictions_LR = lr_model.predict(tfidf_test_x)
time_taken = time.time() - t
print(f'Time taken to predict all values from test set (LR): {time_taken} s.')

print("LR Accuracy Score -> ", accuracy_score(predictions_LR, test_y) * 100)



# OUTPUT of experiments:

# Time taken to train Naive Bayes classifier: 0.0039615631103515625 s.
# Time taken to predict all values from test set (Naive Bayes): 0.0020360946655273438 s.
# Naive Bayes Accuracy Score ->  82.96666666666667

# Time taken to train SVM classifier: 30.287155389785767 s.
# Time taken to predict all values from test set (SVM): 9.524910926818848 s.
# SVM Accuracy Score ->  83.6

# Time taken to train LR classifier: 0.13779664039611816 s.
# Time taken to predict all values from test set (LR): 0.014794111251831055 s.
# LR Accuracy Score ->  83.39999999999999

# As we can see, three classification algorithms performed equally on same task.
# We can see that SVM's accuracy a little bit higher, but time taken to train the model and predict values is much higher too.
# Naive Bayes algorithm is faster to train and predict values, but has the lowest accuracy.