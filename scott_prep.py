import unicodedata
import re
import json

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer

from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
import pandas as pd
import acquire

#######################  Prepare  ###########################

def basic_clean(text):
    '''
    take in a string and apply some basic text cleaning to it:
    * Lowercase everything
    * Normalize unicode characters
    * Replace anything that is not a letter, number, whitespace or a single quote.
    '''
    text = text.lower()  # Lowercase everything
    tedfxt = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')  # Normalize unicode characters
    text = re.sub(r"[^a-z0-9\s']", ' ', text)  # Replace anything that is not a letter, number, whitespace, or single quote
    return text

def tokenize(text):
    '''
    take in a string and tokenize all the words in the string
    '''
    tokenizer = ToktokTokenizer()
    return tokenizer.tokenize(text)


def stemmer(text):
    '''
    accept some text and return the text after applying stemming to all the words
    '''
    stemmer = nltk.stem.PorterStemmer()
    return ' '.join([stemmer.stemmer(word) for word in text.split()])


def lemmatize(text):
    '''
    accept some text and return the text after applying lemmatization to each word
    '''
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])


def remove_stopwords(text, extra_words=[], exclude_words=[]):
    '''
    accept some text and return the text after removing all the stopwords.
    This function defines two optional parameters, extra_words and exclude_words. These parameters define any additional stop words to include,
    and any words that we don't want to remove.
    '''
    ADDITIONAL_STOPWORDS = ['azure','http','com','github','microsoft']

    stopword_list = stopwords.words('english')+ADDITIONAL_STOPWORDS
    for word in extra_words:
        stopword_list.append(word)
    for word in exclude_words:
        stopword_list.remove(word)
    return ' '.join([word for word in text.split() if word not in stopword_list])

############## Wrangle  ###########################
def wrangle_git():
    git_df = pd.read_csv('github.csv',index_col=0)
    git_df = pd.DataFrame(git_df)
    git_df['clean'] = git_df['readme_contents'].apply(basic_clean).apply(tokenize).apply(lambda x: ' '.join(x))
    #stemmer = PorterStemmer()
    #git_df['stemmed'] = git_df['clean'].apply(tokenize).apply(lambda x: [stemmer.stem(word) for word in x]).apply(lambda x: ' '.join(x))
    lemmatizer = WordNetLemmatizer()
    git_df['lemmatized'] = git_df['clean'].apply(tokenize).apply(lambda x: [lemmatizer.lemmatize(word) for word in x]).apply(lambda x: ' '.join(x))
    return git_df


def wrangle_news():
    news_df = acquire.get_news_articles()
    news_df = pd.DataFrame(news_df)
    #news_df = news_df.rename(columns={'content': 'original'})
    #news_df['clean'] = news_df['original'].apply(basic_clean).apply(tokenize).apply(lambda x: ' '.join(x))
    #stemmer = PorterStemmer()
    #news_df['stemmed'] = news_df['clean'].apply(tokenize).apply(lambda x: [stemmer.stem(word) for word in x]).apply(lambda x: ' '.join(x))
    lemmatizer = WordNetLemmatizer()
    news_df['lemmatized'] = news_df['clean'].apply(tokenize).apply(lambda x: [lemmatizer.lemmatize(word) for word in x]).apply(lambda x: ' '.join(x))
    return news_df

