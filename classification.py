import nltk
import math
import numpy as np
from nltk.corpus import twitter_samples
from nltk.tokenize import TweetTokenizer
from gensim.models import Word2Vec

tweet_tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

pos_tokens = []
neg_tokens = []
for tweet in twitter_samples.strings('positive_tweets.json'):
    pos_tokens += tweet_tokenizer.tokenize(tweet)
for tweet in twitter_samples.strings('negative_tweets.json'):
    neg_tokens += tweet_tokenizer.tokenize(tweet)

pos_fd = nltk.FreqDist(pos_tokens)
neg_fd = nltk.FreqDist(neg_tokens)
print('number of positive types:', pos_fd.B())
print('number of positive tokens:', pos_fd.N())
print('number of negative types:', neg_fd.B())
print('number of negative tokens:', neg_fd.N())

def bayes_classifier(filename):
    output_file = open('predictions.txt', 'w', encoding='utf-8')
    num_pos_predictions = 0
    num_neg_predictions = 0

    k = (pos_fd + neg_fd).B()
    pos_count = len(twitter_samples.strings('positive_tweets.json'))
    neg_count = len(twitter_samples.strings('negative_tweets.json'))
    log_prior_pos = math.log(pos_count / (pos_count + neg_count))
    log_prior_neg = math.log(neg_count / (pos_count + neg_count))

    for tweet in twitter_samples.strings(filename): #fix
        total_log_prob_pos = log_prior_pos
        total_log_prob_neg = log_prior_neg
        tokens = tweet_tokenizer.tokenize(tweet)

        for token in tokens:
            total_log_prob_neg += math.log((neg_fd[token] + 1) / neg_fd.N() + k)
            total_log_prob_pos += math.log((pos_fd[token] + 1) / pos_fd.N() + k)

        if total_log_prob_pos > total_log_prob_neg:
            num_pos_predictions += 1
            print('pos', file=output_file)
        else:
            num_neg_predictions += 1
            print('neg', file=output_file)

    print('\nnumber of positive tweets: ', num_pos_predictions, file=output_file)
    print('number of negative tweets: ', num_neg_predictions, file=output_file)

    
tweets = []
for tweet in twitter_samples.strings('positive_tweets.json'):
    tweets += [tweet_tokenizer.tokenize(tweet)]
for tweet in twitter_samples.strings('negative_tweets.json'):
    tweets += [tweet_tokenizer.tokenize(tweet)]
    
model = Word2Vec(tweets)

# stores the sentence vector of each tweet in training set
train_vec = [] #first 5000 are pos, last 5000 are neg
for tweet in twitter_samples.strings('positive_tweets.json') + twitter_samples.strings('positive_tweets.json'):
    vector = np.zeros(shape=(100,))
    num_tokens = 0
    for token in tweet_tokenizer.tokenize(tweet):
        try:
            vector += model[token]
            num_tokens += 1
        except:
            continue
    vector = vector / num_tokens
    train_vec.append(vector)

def get_cossim(vec1, vec2):
    return (np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

# knn classifier using cosine similarity
# https://rare-technologies.com/word2vec-tutorial/
def knn(inputFilename, outputFilename):
    output_file = open(outputFilename, 'w', encoding='utf-8')
    num_pos_predictions = 0
    num_neg_predictions = 0
    
    for tweet in twitter_samples.strings(inputFilename):
        vector = np.zeros(shape=(100,))
        num_tokens = 0
        for token in tweet_tokenizer.tokenize(tweet):
            try:
                vector += model[token]
                num_tokens += 1
            except:
                continue
        vector = vector / num_tokens
    
    for train_vector in train_vec:
        counter = 0
        highest_cossim = 0
        index = 0
        cossim = get_cossim(vector, train_vector)
        if cossim > highest_cossim:
            most_similar = sim
            index = counter
        counter += 1
        
    if index < 5000:
        print('pos', file=output_file)
    else:
        print('neg', file=output_file)
        
    print('\nnumber of positive tweets: ', num_pos_predictions, file=output_file)
    print('number of negative tweets: ', num_neg_predictions, file=output_file)
