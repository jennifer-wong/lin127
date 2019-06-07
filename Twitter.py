#********************** READ BEFORE USE  *********************************************
#NLTK uses a third party library called Twython for handling twitter and must be installed with 'pip install twython'
#prior to running this code
#The environment variable TWITTER must also be set to the path containing the credentials text file
#This can be done by copying the path the credentials.txt file is in, going to environment variables on your computer
#creating a new User variable named TWITTER and pasting the directory of the text file as the variable path
#When specifying a keyword and number of tweets and then running the code, A JSON file will be created and its location will be specified
#in the terminal.  That file needs to be moved to to the directory with this code and that file name will be used in the classification.py in the two lines where each classification is called.
#You can see this in the code as a JSON file is already specified, just change that/
#the documentation for nltk and twitter can be found at http://www.nltk.org/howto/twitter.html
from nltk.twitter import Twitter, credsfromfile
from pprint import pprint
oauth = credsfromfile()
tw = Twitter()
tw.tweets(keywords='shooting',  limit=10) #prints to terminal 
tw.tweets(keywords='shooting',to_screen=False,  limit=10) #prints to file
