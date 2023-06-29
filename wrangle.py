import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split

import unicodedata
import re
import json
import nltk


from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

####################### Imports ############################
    

    
def basic_clean(string):
    lower_string = string.lower()
    
    normal_string = unicodedata.normalize('NFKD', lower_string)\
    .encode('ascii', 'ignore')\
    .decode('utf-8', 'ignore')
    
    normal_no_chars_string = re.sub(r'[^a-z0-9\s]', ' ', normal_string.replace("'", " "))
    
    return normal_no_chars_string

def tokenize(string):
    ttt = ToktokTokenizer()
    return ttt.tokenize(string, return_str=True)

def stem(string):
    ps = nltk.porter.PorterStemmer()
    
    stems = [ps.stem(word) for word in string.split()]

    stemmed_string = ' '.join(stems)
    
    return stemmed_string

def lemmatize(string):
    wnl = nltk.stem.WordNetLemmatizer()
    
    lemmas = [wnl.lemmatize(word) for word in string.split()]
    
    lemmed_string = ' '.join(lemmas)
    
    return lemmed_string

def remove_stopwords(string, extra_words=[], exclude_words=[]):
    stopword_list = stopwords.words('english')
    
    #Removing words from list
    stopword_list = [word for word in stopword_list if word not in exclude_words]
    
    #Adding words to list
    
    for word in extra_words:
        stopword_list.append(word)
    
    no_stop_words = [word for word in string.split() if word not in stopword_list]
    
    no_stop_string = ' '.join(no_stop_words)
    
    return no_stop_string

def parse_df(df, title='title', content='content'):
    '''Takes in a dataframe with title and content columns and returns a df with title,
    original content, and content as it appears cleaned, stemmed, and lemmatized.'''

    new_df = pd.DataFrame()

    new_df['title'] = df[title]
    new_df['original'] = df[content]
    new_df['clean'] = [remove_stopwords(tokenize(basic_clean(string))) for string in df[content]]
    new_df['stemmed'] = [stem(string) for string in new_df.clean]
    new_df['lemmatized'] = [lemmatize(string) for string in new_df.clean]

    return new_df

def filter_words(text):
    master_list = ['function', 'service', '1', 'sample', 'nuget', 'net', 'sdk', 'doc', 'extension', 'use', 'project ', 'template',\
                    'code', 'using', 'build', 'file', 'md', 'dotnet', 'u', 'package', 'go', 'g', 'resource', 'name', 'version', 'cla',\
                    '0', 'run', 'io', 'ak', 'id', 'powershell', 'doc', 'project', 'issue', 'en', 'repository', 'deployment', \
                    'subscription', 'cloud', 'script', 'action', 'app', 'function', 'container', 'deploy', 'create',  'cli', \
                    'python', 'data', 'client', '9', 'node', 'extension', 'model', 'command', '2', 'node', 'action', \
                    'install', 'npm', 'workflow', 'j', 'api']
    
    return ' '.join([word for word in text.split() if word in master_list])

def clean_repos(df):
    l = ['C#', 'PowerShell', 'Go', 'TypeScript', 'Python', 'JavaScript']

    df.language = np.where(df.language.isin(l), df.language, np.nan)      

    df = df.dropna()

    df.readme_contents = df.readme_contents.apply(basic_clean)\
    .apply(tokenize)\
    .apply(lemmatize)\
    .apply(remove_stopwords, extra_words=['azure','http','com','microsoft','github','service','resource','function','project','doc'])

    df['readme_contents'] = df['readme_contents'].apply(lambda x: filter_words(x))

    return df

def split_data(df, target):
    '''
    Takes in a DataFrame and returns train, validate, and test DataFrames; stratifies on target argument.
    
    Train, Validate, Test split is: 56%, 24%, 20% of input dataset, respectively.
    '''
    # First round of split (train+validate and test)
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df[target])

    # Second round of split (train and validate)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123, 
                                       stratify=train_validate[target])
    
    print(f'train -> {train.shape}, {round(train.shape[0]*100 / df.shape[0],2)}%')
    print(f'validate -> {validate.shape},{round(validate.shape[0]*100 / df.shape[0],2)}%')
    print(f'test -> {test.shape}, {round(test.shape[0]*100 / df.shape[0],2)}%')
    
    return train, validate, test  