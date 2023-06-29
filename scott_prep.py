import unicodedata
import re
import json
import numpy as np

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



def clean(text):
    '''
    A simple function to cleanup text data.
    
    Args:
        text (str): The text to be cleaned.
        
    Returns:
        list: A list of lemmatized words after cleaning.
    '''
    ADDITIONAL_STOPWORDS = ['azure','http','com','github','microsoft']
    # basic_clean() function from last lesson:
    # Normalize text by removing diacritics, encoding to ASCII, decoding to UTF-8, and converting to lowercase
    text = (unicodedata.normalize('NFKD', text)
             .encode('ascii', 'ignore')
             .decode('utf-8', 'ignore')
             .lower())
    
    # Remove punctuation, split text into words
    words = re.sub(r'[^\w\s]', '', text).split()
    
    
    # lemmatize() function from last lesson:
    # Initialize WordNet lemmatizer
    wnl = nltk.stem.WordNetLemmatizer()
    
    # Combine standard English stopwords with additional stopwords
    stopwords = nltk.corpus.stopwords.words('english') + ADDITIONAL_STOPWORDS
    
    # Lemmatize words and remove stopwords
    cleaned_words = [wnl.lemmatize(word) for word in words if word not in stopwords]
    
    return cleaned_words

############## Wrangle  ###########################

def join_words(text):
    '''
    used by wrangle_git to join cleaned column
    '''
    return ' '.join(text)

def wrangle_git():
    '''
    this function wrangles the github.csv for use in initial exploration
    it applies general data cleanup principles 
    '''
    git_df = pd.read_csv('github.csv',index_col=0)
    git_df = pd.DataFrame(git_df)

    # remove unwanted languages
    l = ['C#', 'PowerShell', 'Go', 'TypeScript', 'Python', 'JavaScript']
    git_df.language = np.where(git_df.language.isin(l), git_df.language, np.nan)
    # drop nulls
    git_df = git_df.dropna()
    #clean and lemmatize
    git_df['clean'] = git_df['readme_contents'].apply(basic_clean).apply(tokenize).apply(lambda x: ' '.join(x))
    lemmatizer = WordNetLemmatizer()
    git_df['lemmatized'] = git_df['clean'].apply(tokenize).apply(lambda x: [lemmatizer.lemmatize(word) for word in x]).apply(lambda x: ' '.join(x))
    # applies clean function and adds column
    git_df['clean_lem']= (git_df.lemmatized.apply(clean))
    git_df.clean_lem = git_df.clean_lem.apply(lambda x : join_words(x))
    git_df['word_count'] = git_df['clean_lem'].apply(lambda x: len(x.split()))
    return git_df





