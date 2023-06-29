######IMPORTS#####

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

# Basics:
import pandas as pd
import numpy as np
import math
import numpy as np
import scipy.stats as stats
import os

# Data viz:
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud

# Sklearn stuff:
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import mutual_info_score
from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split

## Stats
from scipy.stats import kruskal

## Classification Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

# nltk
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer

from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

## local
import wrangle
import acquire

###### VISUALIZATIONS ########

def visualize_repos(train):
    c_words = ' '.join(train[train.language == 'C#'].readme_contents)
    go_words = ' '.join(train[train.language == 'Go'].readme_contents)
    powershell_words = ' '.join(train[train.language == 'PowerShell'].readme_contents)
    typescript_words = ' '.join(train[train.language == 'TypeScript'].readme_contents)
    python_words = ' '.join(train[train.language == 'Python'].readme_contents)
    javascript_words = ' '.join(train[train.language == 'JavaScript'].readme_contents)

    plt.figure(figsize=(15,20))

    plt.suptitle('Number of Word Occurrences Across Languages')

    plt.subplot(321)
    pd.Series(powershell_words.split()).value_counts().head(20).plot.barh(color='deepskyblue', width=.9)
    plt.title('PowerShell')
    plt.ylabel('Word')
    plt.xlabel('Num Occurrences')

    plt.subplot(322)
    pd.Series(c_words.split()).value_counts().head(20).plot.barh(color='deepskyblue', width=.9)
    plt.title('C#')
    plt.ylabel('Word')
    plt.xlabel('Num Occurrences')

    plt.subplot(323)
    pd.Series(typescript_words.split()).value_counts().head(20).plot.barh(color='deepskyblue', width=.9)
    plt.title('TypeScript')
    plt.ylabel('Word')
    plt.xlabel('Num Occurrences')

    plt.subplot(324)
    pd.Series(go_words.split()).value_counts().head(20).plot.barh(color='deepskyblue', width=.9)
    plt.title('Go')
    plt.ylabel('Word')
    plt.xlabel('Num Occurrences')

    plt.subplot(325)
    pd.Series(python_words.split()).value_counts().head(20).plot.barh(color='deepskyblue', width=.9)
    plt.title('Python')
    plt.ylabel('Word')
    plt.xlabel('Num Occurrences')

    plt.subplot(326)
    pd.Series(javascript_words.split()).value_counts().head(20).plot.barh(color='deepskyblue', width=.9)
    plt.title('JavaScript')
    plt.ylabel('Word')
    plt.xlabel('Num Occurrences')

    plt.tight_layout()
    plt.show()

def get_length_viz():
    '''
    this function brings in a bar plot comparing avg. word count and programming language
    '''
    # Bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=acquire.wrangle_git(), x='language', y='word_count')
    plt.xlabel('Programming Language')
    plt.ylabel('Mean Word Count')
    plt.title('Mean Word Count by Programming Language')
    plt.show()
    
    
def get_bigram_bubble():
    '''
    this function brings in a bubble plot for bigrams
    '''
    indf = acquire.wrangle_git()
    # Tokenize the text into words
    reads = ' '.join(indf['clean_lem'])
    tokens = nltk.word_tokenize(reads)

    # Generate word-level bigrams
    bigrams = list(nltk.bigrams(tokens))

    # Calculate the frequency of bigrams
    bigram_freq = nltk.FreqDist(bigrams)

    # Create a DataFrame from the bigram frequencies
    df_bigrams = pd.DataFrame(list(bigram_freq.items()), columns=['Bigram', 'Frequency'])
    # Sort the bigram frequencies in descending order
    sorted_bigrams = sorted(bigram_freq.items(), key=lambda x: x[1], reverse=True)
    
    # Select the top 15 bigrams
    top_15_bigrams = sorted_bigrams[:15]
    
    # Create a DataFrame for the top 15 bigrams
    df_top_15_bigrams = pd.DataFrame(top_15_bigrams, columns=['Bigram', 'Frequency'])
    
    # Create the bubble chart using Plotly Express
    fig = px.scatter(df_top_15_bigrams, x= df_top_15_bigrams.index, y='Frequency', size='Frequency', hover_data=[df_top_15_bigrams.index],
                     hover_name='Bigram',text='Bigram',template='plotly_white', color='Frequency', size_max=45,
                     labels={'Frequency': 'Frequency', 'Bigram': 'Bigram'}, title='Top 15 Bigram Frequency',
                     color_continuous_scale=px.colors.sequential.Sunsetdark)
    fig.update_layout(height=800)  # Adjust the height of the figure
    fig.update_layout(width=1200)  # Adjust the height of the figure
    
    fig.update_traces(textposition='top center', text=df_top_15_bigrams['Bigram'])  # Add labels to the bubbles
    
    fig.show()
    


def get_trigram_cloud():
    '''
    this function brings in a word cloud for trigrams
    '''
    indf = acquire.wrangle_git()
    # Tokenize the text into words
    reads = ' '.join(indf['clean_lem'])  # Assuming 'indf' is defined somewhere

    tokens = nltk.word_tokenize(reads)

    # Generate word-level trigrams
    trigrams = list(nltk.trigrams(tokens))

    # Calculate the frequency of trigrams
    trigram_freq = nltk.FreqDist(trigrams)

    # Sort the trigram frequencies in descending order
    sorted_trigrams = sorted(trigram_freq.items(), key=lambda x: x[1], reverse=True)

    # Select the top 25 trigrams
    top_25_trigrams = sorted_trigrams[:25]

    # Create a dictionary for the top 25 trigrams
    data = {f'{k[0]}_{k[1]}_{k[2]}': v for k, v in top_25_trigrams}

    # Generate the word cloud from trigram frequencies
    img = WordCloud(background_color='white', width=1200, height=400).generate_from_frequencies(data)

    # Display the word cloud
    plt.figure(figsize=(12, 4))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
###### STATS ########



def word_count_comparison(indf, language_column, word_count_column):
    '''
    Perform Kruskal-Wallis H test to compare word counts across different programming languages.
    
    Parameters:
        indf (DataFrame): Input DataFrame containing the data.
        language_column (str): Name of the column containing the programming languages.
        word_count_column (str): Name of the column containing the word counts.
        
    Returns:
        tuple: Tuple containing the test statistic and p-value.
    '''
    group1='language'
    group2='word_count'
    alpha = .05
    indf= acquire.wrangle_git()
    groups = []
    for language in indf[language_column].unique():
        group = indf[indf[language_column] == language][word_count_column]
        groups.append(group)
    
    # Perform Kruskal-Wallis H test
    statistic, p_value = kruskal(*groups)
    
    print("Kruskal-Wallis H Test Statistic:", statistic)
    print("p-value:", p_value)
    # Evaluate results
    if p_value < alpha:
        print(f'Since the p-value is less than alpha, there exists some relationship between {group1} and the {group2}.\n Therefore, we reject the Ho')
    else:
        print(f'Since the p-value is less than alpha, there is not a significant relationship between {group1} and {group2}.\n Therefore, we fail to reject the Ho')


    return statistic, p_value











######## modeling ###########


def get_baseline(y_train):
    '''
    this function returns a baseline for accuracy
    '''
    baseline_prediction = y_train.mode()
    # Predict the majority class in the training set
    baseline_pred = [baseline_prediction] * len(y_train)
    accuracy = accuracy_score(y_train, baseline_pred)
    baseline_results = {'Baseline': [baseline_prediction],'Metric': ['Accuracy'], 'Score': [accuracy]}
    baseline_df = pd.DataFrame(data=baseline_results)
    return baseline_df  

def decision_tree(X_bow, X_validate_bow, y_train, y_validate):
    """
    This function trains a decision tree classifier on the provided training data, and evaluates its performance on the
    validation data for different values of the 'max_depth' hyperparameter. It then generates a plot of the training and
    validation accuracy scores as a function of 'max_depth', and returns a DataFrame containing these scores.
    Parameters:
    - X_train (pandas.DataFrame): A DataFrame containing the features for the training data.
    - X_validate (pandas.DataFrame): A DataFrame containing the features for the validation data.
    - y_train (pandas.Series): A Series containing the target variable for the training data.
    - y_validate (pandas.Series): A Series containing the target variable for the validation data.
    Returns:
    - scores_df (pandas.DataFrame): A DataFrame containing the training and validation accuracy scores, as well as the
      difference between them, for different values of the 'max_depth' hyperparameter.
    """
    # get data
    scores_all = []
    for x in range(1,20):
        tree = DecisionTreeClassifier(max_depth=x, random_state=123)
        tree.fit(X_bow, y_train)
        train_acc = tree.score(X_bow,y_train)
        val_acc = tree.score(X_validate_bow, y_validate)
        score_diff = train_acc - val_acc
        scores_all.append([x, train_acc, val_acc, score_diff])
    scores_df = pd.DataFrame(scores_all, columns=['max_depth', 'train_acc','val_acc','score_diff'])
    # Plot the results
    sns.set_style('whitegrid')
    plt.plot(scores_df['max_depth'], scores_df['train_acc'], label='Train score', marker='o')
    plt.plot(scores_df['max_depth'], scores_df['val_acc'], label='Validation score', marker='o')
    plt.fill_between(scores_df['max_depth'], scores_df['train_acc'], scores_df['val_acc'], alpha=0.2, color='gray')
    plt.xlabel('Max depth')
    plt.ylabel('Accuracy')
    plt.title('Decision Tree Classifier Performance')
    plt.legend()
    plt.show()
    return scores_df

def random_forest_scores(X_bow, y_train, X_validate_bow, y_validate):
    """
    Trains and evaluates a random forest classifier with different combinations of hyperparameters. The function takes in
    training and validation datasets, and returns a dataframe summarizing the model performance on each combination of
    hyperparameters.
    Parameters:
    -----------
    X_train : pandas DataFrame
        Features of the training dataset.
    y_train : pandas Series
        Target variable of the training dataset.
    X_validate : pandas DataFrame
        Features of the validation dataset.
    y_validate : pandas Series
        Target variable of the validation dataset.
    Returns:
    --------
    df : pandas DataFrame
        A dataframe summarizing the model performance on each combination of hyperparameters.
    """
    #define variables
    train_scores = []
    validate_scores = []
    min_samples_leaf_values = [1, 2, 3, 4, 5, 6, 7, 8 , 9, 10]
    max_depth_values = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    for min_samples_leaf, max_depth in zip(min_samples_leaf_values, max_depth_values):
        rf = RandomForestClassifier(min_samples_leaf=min_samples_leaf, max_depth=max_depth,random_state=123)
        rf.fit(X_bow, y_train)
        train_score = rf.score(X_bow, y_train)
        validate_score = rf.score(X_validate_bow, y_validate)
        train_scores.append(train_score)
        validate_scores.append(validate_score)
    # Calculate the difference between the train and validation scores
    diff_scores = [train_score - validate_score for train_score, validate_score in zip(train_scores, validate_scores)]
    #Put results into a dataframe
    df = pd.DataFrame({
        'min_samples_leaf': min_samples_leaf_values,
        'max_depth': max_depth_values,
        'train_score': train_scores,
        'validate_score': validate_scores,
        'diff_score': diff_scores})
    # Set plot style
    sns.set_style('whitegrid')
    # Create plot
    plt.plot(max_depth_values, train_scores, label='Train score', marker='o')
    plt.plot(max_depth_values, validate_scores, label='Validation score', marker='o')
    plt.fill_between(max_depth_values, train_scores, validate_scores, alpha=0.2, color='gray')
    plt.xticks([2,4,6,8,10],['Leaf 9 and Depth 2','Leaf 7 and Depth 4','Leaf 5 and Depth 6','Leaf 3 and Depth 8','Leaf 1and Depth 10'], rotation = 45)
    plt.xlabel('min_samples_leaf and max_depth')
    plt.ylabel('Accuracy')
    plt.title('Random Forest Classifier Performance')
    plt.legend()
    plt.show()
    return df



############### Initial explore  ###################















