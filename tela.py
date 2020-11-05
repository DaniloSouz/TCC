import PySimpleGUI as sg
import tweepy as tw
import csv
import re
import pandas as pd
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tkinter import messagebox

def clean(text):
        # Passando para letra minuscula
        text = text.lower()
    
        # Removendo tags em HTML
        text = re.sub(r'<[^>]*>', '', text)
    
        # Removendo o nome do usuário do twitter
        text = re.sub(r'@[A-Za-z0-9]+','',text)
    
        # Removendo URL
        text = re.sub('https?://[A-Za-z0-9]','',text)
    
        # Removendo os números
        text = re.sub('[^a-zA-Z]',' ',text)

        # Removendo quebras de linhas
        text = text.rstrip('\n')

        word_tokens = word_tokenize(text)
    
        filtered_sentence = []
        for word_token in word_tokens:
            if word_token not in stop_words:
                filtered_sentence.append(word_token)
    
        # Concatenando
        text = (' '.join(filtered_sentence))
        return text

df = pd.read_csv('TweetsCovid2.csv')
df = df.fillna('')
df ['text'] = df ['text'] + ''
df = df[df['label']!='']

np.array(['Fake', 'TRUE', 'fake'], dtype=object)

df.loc[df['label'] == 'fake', 'label'] = 'FAKE'
df.loc[df['label'] == 'Fake', 'label'] = 'FAKE'

no_of_fakes = df.loc[df['label'] == 'FAKE'].count()[0]
no_of_trues = df.loc[df['label'] == 'TRUE'].count()[0]


stop_words = set(stopwords.words('portuguese'))
  

df['text'] = df['text'].apply(clean)


class TelaPrincipal:
    def __init__(self):
        #Layout
        sg.theme('NeutralBlue')
        layout = [
                    [sg.Text("Informe o texto que deseja avaliar", text_color='white')],
                    [sg.Multiline(size=(40, 10),key='texto')],
                    [sg.Button('Avaliar',  key='avaliar', size=(5,1)), sg.Button('Sair', key='sair', size=(5,1))]
                ]
        #Janela
        janela = sg.Window("Detector de Fake News",layout, element_justification='c', icon='covid.jpg', no_titlebar='true')
        # Extrair os dados da janela
        self.button, self.values = janela.Read()
        
    def Iniciar(self):
        if self.button == 'sair':
            tela.fechar()
        if self.button == 'avaliar':
            textoavaliado = self.values['texto']
            if ('covid' in textoavaliado or 'corona' in textoavaliado or 'quarentena' in textoavaliado or 
                'Quarentena' in textoavaliado or 'Corona' in textoavaliado or 'Covid' in textoavaliado
            ):
                tela.validar(textoavaliado)
            else:
                messagebox.showwarning("Erro", "Palavra Chave inválida")

    def fechar(self):
        sg.WIN_CLOSED

    def validar(self,texto):
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(df['text'].values)
        X = X.toarray()

        y = df['label'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=11)

        clf = MultinomialNB()
        clf.fit(X_train, y_train)
        y = "\nPrecisão nos dados de treino: " + str(round(clf.score(X_train, y_train)*100, 2))
        h = "Precisão nos dados de teste: " + str(round(clf.score(X_test, y_test)*100, 2))
        sentence = texto
        sentence = clean(sentence)
        vectorized_sentence = vectorizer.transform([sentence]).toarray()
        j = "\n\nO texto digitado é possívelmente: " + str(clf.predict(vectorized_sentence))
        j = re.sub(r"[']", "", j)
        c = j.replace("[","")
        k = c.replace("]","")
        messagebox.showinfo("Detector de Fake News", h + y + k)


tela = TelaPrincipal()
tela.Iniciar()