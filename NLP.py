#!/usr/bin/env python
# coding: utf-8

# In[27]:
file1 = open(r'NLP/price.txt', 'r')
lines = file1.readlines()
file2 = open(r'NLP/side effects.txt', 'r')
lines = lines + file2.readlines()
file3 = open(r'NLP/alternative.txt', 'r')
lines = lines + file3.readlines()
file4 = open(r'NLP/active.txt', 'r')
text_list = lines + file4.readlines()
labels = [0 for j in range(100)] + [1 for j in range(100)] + [2 for j in range(100)] + [3 for j in range(100)]

# In[28]:


import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import pandas
import spacy

nlp = spacy.load('en_core_web_sm')

# In[29]:


dataset = pandas.read_csv(r'NLP/final_dataset.csv')


# In[30]:


def preprocess(text_list):
    new_list = []
    for text in text_list:
        t = re.sub("Drug Name", '', text)
        t = re.sub('[?\n]', '', t)
        t = re.sub('[^a-zA-Z\\s]', '', t)
        t = re.sub(r'\s+', ' ', t)
        new_list.append(t)
    return new_list


# In[31]:


class QAModel:
    def __init__(self, df):
        self.vectorizer = TfidfVectorizer()
        self.model = LogisticRegression()
        self.df = df
        self.mp = {0: 'price', 1: 'side effects', 3: 'active'}

    def preprocess(self, text_list):
        new_list = []
        for text in text_list:
            t = re.sub("Drug Name", '', text)
            t = re.sub('[?\n]', '', t)
            t = re.sub('[^a-zA-Z\\s]', '', t)
            t = re.sub(r'\s+', ' ', t)
            new_list.append(t)
        return new_list

    def vectorize(self, text):
        return self.vectorizer.transform(text).toarray()

    def fit(self, text_list, labels):
        clean_texts = self.preprocess(text_list)
        X = self.vectorizer.fit_transform(text_list).toarray()
        self.model.fit(X, labels)

    def classify(self, text):
        text = self.preprocess(text)
        vec = self.vectorizer.transform(text).toarray()
        return self.model.predict(vec)

    def extract_name(self, text):
        # Process the text with spaCy
        doc = nlp(text)
        for name in doc:
            if name.text in self.df['name'].values:
                return name.text
        # Extract drug names (entities labeled as DRUG)
        drug_names = []
        for ent in doc.ents:
            drug_names.append(ent.text)

        if len(drug_names) == 0:
            return doc.text[0]
        return drug_names[0]

    def lcs_length(self, s1, s2):
        m = len(s1)
        n = len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    def find_nearest_name(self, given_name, name_list):
        given_name = str(given_name).lower()
        closest_name = None
        max_lcs_length = 0

        for name in name_list:
            length = self.lcs_length(given_name.lower(), str(name).lower())  # Calculate LCS length
            if length > max_lcs_length:
                max_lcs_length = length
                closest_name = name

        return closest_name

    def find_same_active(self, target_name):
        # Find the active status of the target name
        target_status = self.df[self.df['name'] == target_name]['active'].values[0]

        # Find another name with the same active status
        same_status_names = self.df[self.df['active'] == target_status]['name'].tolist()

        # Remove the target name from the list
        same_status_names.remove(target_name)

        # Return another name with the same active status, if exists
        if same_status_names:
            return f"""{same_status_names[0]} is considered as an alternative for {target_name} \nConsult a healthcare provider before switching medications, especially if you have underlying health conditions or are taking other medications
                """
        else:
            return f'There is no alternatives for {target_name}'

    def responde(self, prompt):
        clas = self.classify([prompt])[0]
        name = self.find_nearest_name(self.extract_name(prompt), self.df['name'].values)
        row = self.df[self.df['name'] == name].iloc[0]
        if clas == 2:
            return self.find_same_active(name)
        else:
            if clas == 0:
                return f"the current price for {name} is about {row[self.mp[clas]]} EGP"
            elif clas == 1:
                return row[self.mp[clas]]
            elif clas == 3:
                return f"the active or main ingredients of {name} is {row[self.mp[clas]]}"
            return

        # In[32]:


model = QAModel(dataset)
model.fit(text_list, labels)


# In[33]:


def extract_first_word(text):
    # Split the text into words
    words = text.split()

    # Return the first word (if it exists)
    if words:
        return words[0]
    else:
        return " "


# In[34]:


names_list = dataset['name'].values
