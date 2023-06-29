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


# Sklearn stuff:
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import mutual_info_score
from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split

## Regression Models

## Classification Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

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


###### STATS ########












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















