import pandas as pd
import tweepy as tw
import csv

auth = tw.OAuthHandler("5ghCjdJ3WMmrpKqcRVh7ZJiQr", "r0XlKhQQ2GZFtM8L9RU7lxc2qebAjetZugo3o9ZHtMIdJLJDr8")
auth.set_access_token("1177476709-nUPM1SoN4lv8I1qJTELGeduWezuKgN2DGovfn9y", "BGzoLW9noV8txGij56D5TmRqzRB9oBrHPY2V3B0TTsRyM")

api = tw.API(auth)

public_tweets = api.home_timeline()

csvFile = open('TweetsCovid.csv', 'a', encoding="utf-8-sig")
csvWriter = csv.writer(csvFile)
query_search = "#covid OR #quaretena OR #corona" + " -filter:retweets"

for tweet in tw.Cursor(api.search,q=query_search,lang="pt", since="2020-10-01").items(100):
    print (tweet.text)
    csvWriter.writerow([tweet.text])