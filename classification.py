import nltk
import math
import json
import numpy as np
from nltk.corpus import twitter_samples
from nltk.tokenize import TweetTokenizer
from gensim.models import Word2Vec

tweet_tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)      #A tokenzier for tweets that shortens words like 'heeeyyyyy' to 'hey', removes upper cases, and removes twitter handles

pos_tokens = []
neg_tokens = []
for tweet in twitter_samples.strings('positive_tweets.json'):       #access the positive and negative twitter samples in the nltk twitter corpus
    pos_tokens += tweet_tokenizer.tokenize(tweet)
for tweet in twitter_samples.strings('negative_tweets.json'):
    neg_tokens += tweet_tokenizer.tokenize(tweet)

pos_fd = nltk.FreqDist(pos_tokens)              #create Frequency distributions for the words
neg_fd = nltk.FreqDist(neg_tokens)
print('number of positive types:', pos_fd.B())
print('number of positive tokens:', pos_fd.N())
print('number of negative types:', neg_fd.B())
print('number of negative tokens:', neg_fd.N())


def bayes_classifier(filename): #Function for the bayes classifier with add 1 smoothing for the specified file of tweets
    output_file = open('shooting.txt', 'w', encoding='utf-8')
    num_pos_predictions = 0
    num_neg_predictions = 0

    k = (pos_fd + neg_fd).B() #total number of bins
    pos_count = len(twitter_samples.strings('positive_tweets.json')) #get the number of positive tokens
    neg_count = len(twitter_samples.strings('negative_tweets.json'))  #number of negative tokens
    log_prior_pos = math.log(pos_count / (pos_count + neg_count))       #logs for very small probabilities
    log_prior_neg = math.log(neg_count / (pos_count + neg_count))

    tweets = []
    with open(filename) as f: #open the specified json file containing tweets and load each line as a seperate tweet
        for line in f:
            tweets.append(json.loads(line))

    for tweet in tweets:  # for each tweet perform the sentiment analysis
        total_log_prob_pos = log_prior_pos
        total_log_prob_neg = log_prior_neg
        tokens = tweet_tokenizer.tokenize(tweet["text"]) #get the raw tweet text from each tweet

        for token in tokens:
            total_log_prob_neg += math.log((neg_fd[token] + 1) / neg_fd.N() + k) #bayes formula for pos/neg probability
            total_log_prob_pos += math.log((pos_fd[token] + 1) / pos_fd.N() + k)

        if total_log_prob_pos > total_log_prob_neg: #if it is more likely to be positive
            num_pos_predictions += 1
            print('pos', file=output_file)      #record to output file
        else:
            num_neg_predictions += 1
            print('neg', file=output_file)

    print('\nnumber of positive tweets: ', num_pos_predictions, file=output_file)
    print('number of negative tweets: ', num_neg_predictions, file=output_file)
bayes_classifier('tweets.20190606-143647.json')

tweets = []
for tweet in twitter_samples.strings('positive_tweets.json'):
    tweets += [tweet_tokenizer.tokenize(tweet)]
for tweet in twitter_samples.strings('negative_tweets.json'):
    tweets += [tweet_tokenizer.tokenize(tweet)]

model = Word2Vec(tweets)

# stores the sentence vector of each tweet in training set
train_vec = []  # first 5000 are pos, last 5000 are neg
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

    tweets = []
    with open(inputFilename) as f:  # open the specified json file containing tweets and load each line as a seperate tweet
        for line in f:
            tweets.append(json.loads(line))

    for tweet in tweets:
        vector = np.zeros(shape=(100,))
        num_tokens = 0
        for token in tweet_tokenizer.tokenize(tweet["text"]):
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
                most_similar = highest_cossim
                index = counter
            counter += 1

        if index < 5000:
            num_pos_predictions += 1
            print('pos', file=output_file)
        else:
            num_neg_predictions += 1
            print('neg', file=output_file)

    print('\nnumber of positive tweets: ', num_pos_predictions, file=output_file)
    print('number of negative tweets: ', num_neg_predictions, file=output_file)

knn('tweets.20190606-143647.json', 'shooting2.txt')
