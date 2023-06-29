# NLP-Project
### Wilson Velasco and Scott Barnett
### June 30, 2023
## Project description with goals
### Description
* We want to be able to predict the coding langage used in the repository 

### Goals¶
* Construct a ML model that predicts 
* Find key indicatorts of coding language 
* Deliver a report to the data science team 
* Deliver a presentation of findings to a general audience
## Initial hypotheses and/or questions you have of the data, ideas
There should be some combination of features that can be used to build a predictive model for 
* 1. 
* 2. 
* 3. 
* 4. 
*****************************************
## Project Plan 
* Data acquired from GitHub
    * 
    * 
* Prepare data
    * Took value_counts() of each language. Limited dataset to six most popular languages.
    * Removed rows with nulls in language columns, since language is our target variable.
    * Processed text in readme_contents to clean, tokenize, lemmatize, and remove stopwords from text.
    * Found top 20 words in each language prior to split, and removed all other words.
    * Outliers were not addressed in this itteration
    * Split data into **train**, **validate**, **test**       
## Explore data in search of drivers of 
* Answer the following initial questions
    * 1. Are there any key words that are used more often in certain languages?
    * 2. Is the length (word count) of the readme file related to the programing language used?
    * 3. Could the top 15 bigrams be used to predict language?
    * 4. Could the top 25 trigrams be used to predict language?
* Develop a model to predict programing lanuage used in the Github repository
    * Run the models 
    * Select best model
* Draw conclusions

## Data Dictionary
| Feature | Datatype | Key | Definition |
|---|---|---|---|
| repo | object | unique | Name of GitHub directory |
| language | object | unique | Programming language used |
| readme_contents | object | unique  | contents of the readme file  |

## Steps to Reproduce
* 1. Data is collected from 
    * 
    * 
    * 
* 2. Clone this repo.
* 3. Put the data in the file containing the cloned repo.
* 4. Run notebook.

## Takeaways and Conclusions
* The four features we evaluated closely: 
* 
* 
* 
  
  
* Our best model was 
    * It returned an `fill` score of `fill' out performing the baseline by more than `fill`

# Recommendations
* The model provided can be used to create an 
# Next Steps
* If provided more time to work on the project we would want to explore 