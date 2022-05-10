import pandas as pd
from sklearn.preprocessing import FunctionTransformer

pd.set_option('display.max_columns', 25)
pd.set_option('display.max_rows', 50)
import numpy as np
np.random.seed(3301)
import pandas as pd

from sklearn.model_selection import train_test_split


import nltk


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




df_eleg = pd.read_csv('./Datos/ElegibilidadEstudiantes/clinical_trials_on_cancer_data_clasificacion.csv', sep=',', encoding='utf-8', dtype='unicode')

def preprocessor(df):
    df= df.replace('""', '')
    df = df.str.strip(' ')
    df = df.str.split('.').str[1]
    return df
pre = [('preproc', FunctionTransformer(preprocessor))]
def encoder(df):
    df.loc[df['label'] == '__label__0', 'label'] = 0
    df.loc[df['label'] == '__label__1', 'label'] = 1
encoder(df_eleg)

porter = PorterStemmer()
stop = stopwords.words('english')
def tokenizer_porter(sentence):
    tokens = sentence.split()
    stemmed_tokens = [porter.stem(token) for token in tokens if token not in stop]
    return ' '.join(stemmed_tokens)

def transformer_tokenizer(df):
    df = df.apply(tokenizer_porter)
    return df

pre += [('porter', FunctionTransformer(transformer_tokenizer))]
tfidf = TfidfVectorizer( ngram_range= (1,1), use_idf= True)
pre += [('tfidf', tfidf)]
Y = df_eleg['label'].astype('int')
X = df_eleg['study_and_condition']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.15, random_state = 98572398)
model = [('CNB',MultinomialNB(alpha= 1.1, fit_prior= True))]


p1 = Pipeline(pre+model)

p1.fit(X_train, Y_train)


joblib.dump(p1,"naivebayes.joblib", compress=0, protocol=None, cache_size=None)


model2 = [('clf',LogisticRegression(random_state=0, C=1.0, penalty='l2', solver='newton-cg'))]

p2 = Pipeline(pre+model2)

p2.fit(X_train, Y_train)
joblib.dump(p2, "logisticregression.joblib", compress=0, protocol=None, cache_size=None)

model3 = [('SVM', svm.SVC(C=1, gamma=1, kernel='poly'))]
pipeline = Pipeline(pre + model3)

pipeline.fit(X_train, Y_train)
joblib.dump(pipeline, "SVM.joblib", compress=0, protocol=None, cache_size=None)