import nltk
lemmatizer = nltk.wordnet.WordNetLemmatizer()
import pandas as pd
import numpy as np
from collections import Counter
import re
import csv
import io
import datetime

pth = r"C:\Users\patrik\Desktop\EPFL\ada" #change location according to your nltk data path
nltk.data.path.append(pth)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

################################# PREPROCESSING ###########################################

data_folder = 'twitter-datasets/'

def read_data(filepath):
    tweets       = []
    tweets_file  = io.open(data_folder + filepath, 'r', encoding='utf8')

    for line in tweets_file:
        tw = line.strip()
        tweets.append(tw)
    tweets = pd.DataFrame(tweets, columns=['tweet'])

    return process(tweets)

def combine_pos_neg_tweet(pos_file, neg_file):
    train_pos = read_data(pos_file)
    train_pos['target'] = 1
    train_neg = read_data(neg_file)
    train_neg['target'] = -1
    train_data = pd.concat([train_neg, train_pos]) #merge positive and negative tweets
    train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True) #shuffle the datas

    print('we have {} positive tweets, and {} negative ones.'.format(len(train_pos), len(train_neg)))

    return train_data

def remove_contraction(tweets):

    contractions_dict = {
        '\'s': '', 'who\'s': 'who is', 'i\'m': 'i am', 'n\'t': 'n not', 'why\'s': 'why is', 'he\'s': 'he is', '\'ll': ' will',
        '\'l': ' will', 'what\'s': 'what is', 'when\'s': 'when is', '\'re': ' are', '\'ve': ' have',
        '\'d': ' would', 'how\'s': 'how is', 'it\'s': 'it is', 'that\'s': 'that is', 's\'': '',
    }

    pat = re.compile('|'.join(contractions_dict.keys()))
    tweets =re.sub(r'<user>','',tweets)
    tweets = re.sub(r'<url>','',tweets)
    tweets = re.sub(r'rt','',tweets)
    tweets = pat.sub(lambda x: contractions_dict[x.group()], tweets)

    return tweets

def clean_redundant_character(tweet):
    cleaned_tweet = ''
    words = tweet.split()

    for word in words:
        cleaned_tweet += re.sub(r'([a-z])\1+$', r'\1 <redundant>' , word) + ' '

    return cleaned_tweet.strip() #remove whitespace at end

def handle_hashtag_number(tweet):
    clean_tweet = ''
    words     = tweet.split()

    for w in words:
        try:
            remaining = re.sub('[,\.:%_\+\_\%\*\/\-]', '', w)
            float(remaining)
            clean_tweet += '<number> '
        except:
            if w.startswith("#"):
                clean_tweet += w[1:] + ' <hashtag> '
            else:
                clean_tweet += w + ' '

    return clean_tweet.strip()

def remove_space_punctuation(tweets):
    puncts = ['?', '.', '!', '+', '(', ')']
    clean_tweets = ''
    first = ''
    j = 0
    for w in tweets.split() :
        if(w in puncts) :
            if( first != w) :
                first = w
            else :
                j += 1
                if( j ==1) :
                    clean_tweets += first + ' <repeat> '
        else :
            clean_tweets += w + ' '
    return clean_tweets

def map_sentiment(tweet):

    cleaned_tweet = ''
    words = tweet.split()

    #emoji sentiment
    for word in words:
        if(word in pos_emoji) :
            cleaned_tweet += pos_emoji[word] + ' '
        elif(word in neg_emoji) :
            cleaned_tweet += neg_emoji[word] + ' '
        else :
            cleaned_tweet += word + ' '

    cleaned_tweet = cleaned_tweet.split()

    #word sentiment
    sentiment_tweet = ''
    for word in cleaned_tweet:
        if(word in positive_words) :
            sentiment_tweet += 'positive ' + word + ' '
        elif(word in negative_words) :
            sentiment_tweet += 'negative ' + word + ' '
        else :
            sentiment_tweet += word + ' '
    return sentiment_tweet.strip()

def stemSentence(tweet):
    s = tweet.split()
    s = [lemmatizer.lemmatize(x) for x in s]
    return " ".join(s)

def process(tweets) :
    tweets.tweet = tweets.tweet.apply(lambda t : remove_contraction(t))
    tweets.tweet = tweets.tweet.apply(lambda t : map_sentiment(t))
    tweets.tweet = tweets.tweet.apply(lambda t : handle_hashtag_number(t))
    tweets.tweet = tweets.tweet.apply(lambda t : remove_space_punctuation(t))
    tweets.tweet = tweets.tweet.apply(lambda t : clean_redundant_character(t))
    tweets.tweet = tweets.tweet.apply(lambda t : stemSentence(t))
    return tweets

def log_reg_helper(X_train, y_train, X_test, max_f = None):

    start_time = datetime.datetime.now()
    log = LogisticRegression(max_iter=400, penalty='l2', C=10, tol=1e-4, random_state=42)
    tfidf_vect = TfidfVectorizer(max_features=max_f, ngram_range=(1,3))
    corpus = X_train.append(X_test)
    tfidf_vect.fit(corpus)
    train_tfidf = tfidf_vect.transform(X_train)
    test_tfidf = tfidf_vect.transform(X_test)

    duration = datetime.datetime.now() - start_time

    print('TFIDF: the corpus fitted and tranformed in {} s'.format(duration.total_seconds()))

    log.fit(train_tfidf, y_train)
    predict = log.predict(test_tfidf)

    duration = datetime.datetime.now() - start_time
    print('train finished in {} s'.format(duration.total_seconds()))

    return predict

def to_kaggle(id, predicted):
    ids = id
    with open('logistic_regression_prediction.csv', 'w', newline='') as csvfile:
            fieldnames = ['Id', 'Prediction']
            writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
            writer.writeheader()
            for r1, r2 in zip(ids, predicted):
                writer.writerow({'Id':int(r1),'Prediction':int(r2)})
#########################################################################################################33

neg_emoji = io.open('opinion-lexicon-english/negative_emojis.txt', encoding='utf-8-sig').read().split("\n")
neg_emoji = dict([x.split() for x in neg_emoji])
pos_emoji = io.open('opinion-lexicon-english/positive_emojis.txt', encoding='utf-8-sig').read().split("\n")
pos_emoji = dict([x.split() for x in pos_emoji])
positive_words = set(io.open('opinion-lexicon-english/positive-words.txt', encoding = "ISO-8859-1").read().split())
negative_words = set(io.open('opinion-lexicon-english/negative-words.txt', encoding = "ISO-8859-1").read().split())


start_time = datetime.datetime.now()
kaggle_data = combine_pos_neg_tweet('train_pos_full.txt', 'train_neg_full.txt')
kaggle_test = read_data('test_data.txt')

X_train = kaggle_data.tweet
y_train = kaggle_data.target
X_test = kaggle_data.tweet

print('data preprocessed: X train = {}, y train = {}, X test = {}'.format(np.shape(X_train), np.shape(y_train), np.shape(X_test)))
duration = datetime.datetime.now() - start_time
print('finished preprocess in {}'.format(duration.total_seconds()))

predicted = log_reg_helper(X_train, y_train, X_test)

print('prediction : {}'.format(Counter(predicted)))

to_kaggle(X_test.id.values, predicted)

print('script finished')
