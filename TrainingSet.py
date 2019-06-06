import nltk
nltk.download('twitter_samples')
from nltk.corpus import twitter_samples
from nltk.tokenize import TweetTokenizer
print(twitter_samples.fileids())
tweet_tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
p_tw = twitter_samples.strings('positive_tweets.json')
n_tw = twitter_samples.strings('negative_tweets.json')
pos_count = 5000
neg_count = 5000
pos_tokens = []
neg_tokens = []
for t in p_tw:
    pos_tokens = pos_tokens + tweet_tokenizer.tokenize(t)
for t in n_tw:
    neg_tokens = neg_tokens + tweet_tokenizer.tokenize(t)
p_fd = nltk.FreqDist(pos_tokens)
n_fd = nltk.FreqDist(neg_tokens)