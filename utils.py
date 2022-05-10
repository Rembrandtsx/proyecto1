from sklearn.preprocessing import FunctionTransformer
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

pd.set_option('display.max_columns', 25)
pd.set_option('display.max_rows', 50)
import numpy as np
np.random.seed(3301)
import pandas as pd

from sklearn.model_selection import train_test_split


import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


pd.set_option('display.max_colwidth', None)  # or 199

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
import joblib
from sklearn import svm


porter = PorterStemmer()
stop = stopwords.words('english')

def preprocessor(df):
    df= df.replace('""', '')
    df = df.str.strip(' ')
    df = df.str.split('.').str[1]
    return df

def tokenizer_porter(sentence):
    tokens = sentence.split()
    stemmed_tokens = [porter.stem(token) for token in tokens if token not in stop]
    return ' '.join(stemmed_tokens)

def transformer_tokenizer(df):
    df = df.apply(tokenizer_porter)
    return df