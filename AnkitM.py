import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes

df=pd.read_csv('C:\\Users\\User\\Desktop\\ANkit1\\labeledTrainData.tsv',sep='\t',names=['id','sentiment','review'])
print("\nLength of the datasets={}".format(len(df)))

dupes=df.duplicated()

print("\nTotal number of duplicates data={}".format(sum(dupes)))

df_uniq=df.drop_duplicates()
print("\nTotal number of unique data={}".format(len(df_uniq)))
print("\nHead view of the Datasets:-\n")
print(df_uniq.head())

df_x=df_uniq["review"]
df_y=df_uniq["sentiment"]


x_train,x_test,y_train,y_test=train_test_split(df_x,df_y, test_size=0.2,random_state=42)

cv=TfidfVectorizer(stop_words=set("eng"))

x_traincv=cv.fit_transform(x_train)

x_testcv=cv.transform(x_test)

mnb=naive_bayes.MultinomialNB()
mnb.fit(x_traincv,y_train)
pred=mnb.predict(x_testcv)
actual=np.array(y_test)

print("\nTotal number of test data provided = {} ".format(len(actual)))

count=0
for i in range (len(pred)):
    if pred[i]==actual[i]:
        count+=1
        
print("\nThe number of Correct Prediction by the model = {} ".format(count))
a=count
b=len(actual)
print("\nThe accuracy percentage of the Multinomial Naive Bayes Classifier={}".format((a/b*100)))      
#Test=["The Hotel was very good but the service provided was not upto the mark."]
#Test1=cv.transform(Test)
#prediction=mnb.predict(Test1)
#print(Test)
#print(prediction)


#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

import sys,tweepy,csv,re

class SentimentAnalysis:

    def __init__(self):
        self.tweets = []
        self.tweetText = []

    def DownloadData(self):
        consumerKey ='XXXXXXXXXXXXXXXXXXXXXXXXX'
        consumerSecret = 'XXXXXXXXXXXXXXXXXXXXXX'
        accessToken = 'XXXXXXXXXXXXXXXXXXXXXXXX'
        accessTokenSecret = 'XXXXXXXXXXXXXXXXXXXXXXXXXXX'
        auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
        auth.set_access_token(accessToken, accessTokenSecret)
        api = tweepy.API(auth)

        searchTerm = input("Enter Keyword to search about: ")
        NoOfTerms = int(input("Enter how many tweets to search: "))

        self.tweets = tweepy.Cursor(api.search, q=searchTerm, lang = "en").items(NoOfTerms)

        csvFile = open('Ankit.csv', 'a')

        csvWriter = csv.writer(csvFile)

        for tweet in self.tweets:
            self.tweetText.append(self.cleanTweet(tweet.text).encode('utf-8'))
            csvWriter.writerow(self.tweetText)
            A=np.array(self.tweetText)
            analysis =cv.transform(A)
        X=mnb.predict(analysis)
        print(X)
        countp=0
        countn=0
        for i in range(len(X)):
            if(X[i]=='1'):
                countp+=1
            elif(X[i]=='0'):
                countn+=1
        print("The number of positive tweets are {}.".format(countp))
        print("The number of negative tweets are {}.".format(countn))
        if(countp>countn):
            print("Peoples are thinking Positively upon {}.".format(searchTerm))
        elif(countp==countn):
            print("Peoples are thinking Neutraly upon {}.".format(searchTerm))
        else:
            print("Peoples are thinking Negatively upon {}.".format(searchTerm))
        csvFile.close()

    def cleanTweet(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w +:\ / \ / \S +)", " ", tweet).split())




if __name__== "__main__":
    sa = SentimentAnalysis()
    sa.DownloadData()
