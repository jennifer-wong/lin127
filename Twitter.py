#********************** READ BEFORE USE  *********************************************
#NLTK uses a third party library called Twython for handling twitter and must be installed with 'pip install twython'
#prior to running this code
#The environment variable TWITTER must also be set to the path containing the credentials text file'
#the documentation for nltk and twitter can be found at http://www.nltk.org/howto/twitter.html
from nltk.twitter import Twitter, credsfromfile, 
from pprint import pprint
oauth = credsfromfile()
tw = Twitter()
tw.tweets(keywords='shooting',  limit=10) #prints to terminal 
tw.tweets(keywords='shooting',to_screen=False,  limit=10) #prints to file
