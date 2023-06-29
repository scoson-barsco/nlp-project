# NLP-Project

### Wilson Velasco and Scott Barnett

### June 30, 2023

### Project Description
* Using a list of repositories scraped from GitHub, we want to be able to predicting the programming language used for that repo using their README files. 

### Goals¶
* Construct a ML model that accurately predicts programming language
* Find key indicatorts of coding language 
* Deliver a report to the data science team 
* Deliver a presentation of findings to a general audience

### Data Dictionary

| Feature | Datatype | Key | Definition |
|---|---|---|---|
| repo | object | unique | Name of GitHub directory |
| language | object | unique | Programming language used |
| readme_contents | object | unique  | contents of the readme file  |

## Initial hypotheses and/or questions you have of the data, ideas
There should be some combination of features that can be used to build a predictive model for 
* 1. Are there any key words that are used more often in certain languages?
* 2. Are the readme word counts related to the programming language used?
* 3. Could the top 15 bigrams be used to predict language?
* 4. Coule the top 25 trigrams be used to predict language?

*****************************************
## Project Plan 
* Acquire Data
    * Data scraped from GitHub from Azure organization
        * File cached to local repository
        * Data read into notebook through cached .csv
    
    * Original dataframe contained 846 rows and 3 columns
    * The data was acquired on 27 June 2023
    * Each row represents a seperate GitHub repository
    * Each column represents a feature of the repository

* Prepared data
    * Took value_counts() of each language. 
    * Removed rows with nulls in language columns, since language is our target variable.
    * Processed text in readme_contents to clean, tokenize, lemmatize, and remove stopwords from text.
        * Included "azure," "microsoft," "http," "github," and other words in list of stopwords to remove, as they were common across all languages.
    * Handled outliers
        * Limited dataset to top six most popular languages.
        * Found top 20 words in each language prior to split, and removed all other words.
    * Split data into **train**, **validate**, **test**    
## Explore data in search of drivers of programming language
* Answer the following initial questions
    * 1. Are there any key words that are used more often in certain languages?
    * 2. Is the length (word count) of the readme file related to the programing language used?
    * 3. Could the top 15 bigrams be used to predict language?
    * 4. Could the top 25 trigrams be used to predict language?
* Develop a model to predict programing language used in the Github repository
    * Run the models
    * Select best model
* Conclusions

## Data Dictionary
| Feature | Datatype | Key | Definition |
|---|---|---|---|
| repo | object | unique | Name of GitHub directory |
| language | object | unique | Programming language used |
| readme_contents | object | unique  | contents of the readme file  |

## Steps to Reproduce
* 1. Data is collected from the list of repositories from the Azure organization's page on GitHub
    * <a href='https://github.com/orgs/Azure/repositories?type=all'> Link </a>
* 2. Clone this repo.
* 3. Put the data in the file containing the cloned repo.
* 4. Run notebook (file will run based on cached .csv file containing data).

## Takeaways and Conclusions
* The features we evaluated closely: 
    * could be used to predict a given README's programing language
* Using Bigrams or Trigrams had a negative impact on our model.
* Our best model was Multinomial Naive Bayes.
    * resulted in an **increase in accuracy of 32.7%** or nearly a **2.5 fold increase** over baseline
  
* Our best model was 
    * It returned an accuracy score of 55.3%, outperforming the baseline by more than 30%.

# Recommendations
* The model provided can be used to predict the programming language used.
# Next Steps
* If provided more time to work on the project we would want to explore ways to properly include the word count of the readme files for use in the model
* We removed a significant amount of words from our corpus. Given more time, we would want to reduce the amount of word "outliers" that we exclude from our dataset, and perhaps include the top 1,000 words for each language.
* On the same token, we would add more words to our stopwords list, as there is a significant amount of overlap in terminology across languages. This way, we can pull out uniquely identifying elements of each language and theoretically increase our accuracy.