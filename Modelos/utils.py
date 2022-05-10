from sklearn.preprocessing import FunctionTransformer
import nltk
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
from nltk.corpus import stopwords

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