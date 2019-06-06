import nltk
import math
import numpy as np
from nltk.corpus import twitter_samples
from gensim.models import Word2Vec

pos_tokens = twitter_samples.tokenized('positive_tweets.json')
pos_tokens = [item for sublist in pos_tokens for item in sublist]
pos_fd = nltk.FreqDist(pos_tokens)
print('number of types:', pos_fd.B())
print('number of tokens:', pos_fd.N())
#for word in pos_fd.most_common(30):
#    print(word)

neg_tokens = twitter_samples.tokenized('negative_tweets.json')
neg_tokens = [item for sublist in neg_tokens for item in sublist]
neg_fd = nltk.FreqDist(neg_tokens)
print('number of types:', neg_fd.B())
print('number of tokens:', neg_fd.N())
#for word in pos_fd.most_common(30):
#    print(word)

# bayes classifier
pos_count = len(twitter_samples.strings('positive_tweets.json'))
neg_count = len(twitter_samples.strings('negative_tweets.json'))
k = (pos_fd + neg_fd).B()
log_prior_pos = math.log(pos_count / (pos_count + neg_count))
log_prior_neg = math.log(neg_count / (pos_count + neg_count))
output_file = open('predictions.txt', 'w', encoding='utf-8')

for tweet in twitter_samples.tokenized('tweets.20150430-223406.json'):
    total_log_prob_pos = log_prior_pos
    total_log_prob_neg = log_prior_neg
    
    for token in tweet:
        total_log_prob_neg += math.log((neg_fd[token] + 1) / neg_fd.N() + k)
        total_log_prob_pos += math.log((pos_fd[token] + 1) / pos_fd.N() + k)
        
    if total_log_prob_pos > total_log_prob_neg:
        num_pos_predictions += 1
        print(tweet, 'pos', file=output_file)
    else:
        num_neg_predictions += 1
        print(tweet, 'neg', file=output_file)

# cosine similarity
#https://rare-technologies.com/word2vec-tutorial/
model = Word2Vec(twitter_samples.tokenized('negative_tweets.json') + 
                 twitter_samples.tokenized('positive_tweets.json'))
model.save('twitterModel')

train_vec = [] #first 5000 are pos, last 5000 neg
for tweet in twitter_samples.tokenized('positive_tweets.json') + twitter_samples.tokenized('negative_tweets.json'):
    vector = np.zeros(shape=(100,))
    for token in tweet:
        try:
            vector += model[token]
        except:
            print('token not in vocab')
    train_vec.append(vector)
    
def cos_sim(vec1, vec2):
    return (np.dot(vec1, vec2) / (np.linalg.norm(vec1, 2) * np.linalg.norm(vec2, 2)))
    
output_file = open('predictions2.txt', 'w', encoding='utf-8')
for tweet in twitter_samples.tokenized('tweets.20150430-223406.json'):
    vector = np.zeros(shape=(100,))
    for token in tweet:
        try:
            vector += model[token]
        except:
            print('token not in vocab')
    for train_vector in train_vec:
        counter = 0
        most_similar = 0
        index = 0
        sim = cos_sim(vector, train_vector)
        if sim > most_similar:
            most_similar = sim
            index = counter
        counter += 1
    if index < 5000:
        print(tweet, 'pos', file=output_file)
    else:
        print(tweet, 'neg', file=output_file)
