import pandas as pd
import tweepy as tw
import csv

auth = tw.OAuthHandler("", "")
auth.set_access_token("", "")

api = tw.API(auth)

public_tweets = api.home_timeline()

csvFile = open('TweetsCovid.csv', 'a', encoding="utf-8-sig")
csvWriter = csv.writer(csvFile)
query_search = "#covid OR #quaretena OR #corona" + " -filter:retweets"

for tweet in tw.Cursor(api.search,q=query_search,lang="pt", since="2020-10-01").items(100):
    print (tweet.text)
    csvWriter.writerow([tweet.text])