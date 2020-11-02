import tweepy as tw
import csv
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords



##### Data collection & analysis
df = pd.read_csv('covid_fake_news.csv')
df.head()

df = df.fillna('')

df ['title_text_source'] = df ['title'] + '' + df ['text'] + '' + df ['source'] 
df.head ()

df = df[df['label']!='']


np.array(['Fake', 'TRUE', 'fake'], dtype=object)

df.loc[df['label'] == 'fake', 'label'] = 'FAKE'
df.loc[df['label'] == 'Fake', 'label'] = 'FAKE'

no_of_fakes = df.loc[df['label'] == 'FAKE'].count()[0]
no_of_trues = df.loc[df['label'] == 'TRUE'].count()[0]


stop_words = set(stopwords.words('english'))

def clean(text):
    # Lowering letters
    text = text.lower()
    
    # Removing html tags
    text = re.sub(r'<[^>]*>', '', text)
    
    # Removing twitter usernames
    text = re.sub(r'@[A-Za-z0-9]+','',text)
    
    # Removing urls
    text = re.sub('https?://[A-Za-z0-9]','',text)
    
    # Removing numbers
    text = re.sub('[^a-zA-Z]',' ',text)
    
    word_tokens = word_tokenize(text)
    
    filtered_sentence = []
    for word_token in word_tokens:
        if word_token not in stop_words:
            filtered_sentence.append(word_token)
    
    # Joining words
    text = (' '.join(filtered_sentence))
    return text

df['title_text_source'] = df['title_text_source'].apply(clean)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['title_text_source'].values)
X = X.toarray()

y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=11)

clf = MultinomialNB()
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
#cm recebe a matriz de confusão onde passa os parametros da variável y_test e predections
cm = confusion_matrix(y_test, predictions)

#Matriz confusão inferface apresentando a matriz
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['FAKE', 'TRUE'], yticklabels=['FAKE', 'TRUE'], cmap=plt.cm.Blues, cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
#plt.show()

sentence = 'The Corona virus is a man made virus created in a Wuhan laboratory. Doesn’t @BillGates finance research at the Wuhan lab?'
sentence = clean(sentence)
vectorized_sentence = vectorizer.transform([sentence]).toarray()
#print(clf.predict(vectorized_sentence))


auth = tw.OAuthHandler("5ghCjdJ3WMmrpKqcRVh7ZJiQr", "r0XlKhQQ2GZFtM8L9RU7lxc2qebAjetZugo3o9ZHtMIdJLJDr8")
auth.set_access_token("1177476709-nUPM1SoN4lv8I1qJTELGeduWezuKgN2DGovfn9y", "BGzoLW9noV8txGij56D5TmRqzRB9oBrHPY2V3B0TTsRyM")

api = tw.API(auth)

public_tweets = api.home_timeline()

csvFile = open('TweetsCovid.csv', 'a')
csvWriter = csv.writer(csvFile)
query_search = "#covid OR #quaretena OR #corona" + " -filter:retweets"

for tweet in tw.Cursor(api.search,q=query_search).items(500):
    print (tweet.created_at, tweet.text)
    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])


#for tweet in cursor_tweets:
#    print(tweet.created_at)
#    print(tweet.text)

#twkeys = tweet._json.keys()


#tweets_dict = {}
#tweets_dict = tweets_dict.fromkeys(twkeys)

#or tweet in cursor_tweets:
#    for key in tweets_dict.keys():
 #       try:
  #          twkey = tweet._json[key]
   #         tweets_dict[key].append(twkey)
    #    except KeyError:
     #       twkey = ""
      #      tweets_dict[key].append("")
       # except:
        #    tweets_dict[key].append("")


#dfTweets.to_csv('TweetsCovid.csv', index = 'A')