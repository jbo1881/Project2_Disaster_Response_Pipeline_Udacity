# import libraries
import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine

import re
import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB


def load_data(database_filepath):
    """
    Load data from a SQLite database and prepare it for machine learning.

    Parameters:
        database_filepath (str): Filepath to the SQLite database.

    Returns:
        X (pd.Series): Series containing the messages.
        y (pd.DataFrame): DataFrame containing the target categories.
        category_names (list): List of category names.
    """

    # Create a database engine and read data from the database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('Disaster_Messages', engine)

    # Separate features (messages) and target categories
    X = df['message']
    y = df.loc[:, 'related':]  # Select columns related to target categories

    # Extract the category names for reference
    category_names = list(df.columns[4:])  # Columns 4 and onwards are category names

    return X, y, category_names



def tokenize(text):
    """
    Tokenize and preprocess text for natural language processing.

    This function performs the following steps:
    1. Normalizes the text by converting it to lowercase and removing non-alphanumeric characters.
    2. Tokenizes the normalized text into words.
    3. Removes common English stop words.
    4. Performs lemmatization on the remaining words.

    Parameters:
        text (str): Input text to be tokenized and preprocessed.

    Returns:
        list of str: A list of preprocessed and tokenized words.
    """

    # 1. Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", ' ', text.lower())

    # 2. Tokenize text
    words = word_tokenize(text)

    # 3. Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]

    # 4. Lemmatization
    lemmat = [WordNetLemmatizer().lemmatize(w) for w in words]

    return lemmat



def build_model():
    """
    Build a machine learning model pipeline for text classification using Naive Bayes.

    This function creates a pipeline that includes text vectorization, TF-IDF transformation,
    and a MultiOutputClassifier with Multinomial Naive Bayes as the estimator. It also sets up
    a parameter grid for grid search and cross-validation.

    Returns:
        cv (GridSearchCV): Grid search object for model tuning.
        X_train (pd.Series): Training features.
        X_test (pd.Series): Testing features.
        y_train (pd.DataFrame): Training target categories.
        y_test (pd.DataFrame): Testing target categories.
    """

    # Pipeline: Naive Bayes classifier
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf',  MultiOutputClassifier(MultinomialNB()))
        ])
    
    # Define the parameter grid to search over for Naive Bayes
    parameters = {
        'tfidf__use_idf': [True, False],  # TF-IDF use_idf parameter
        'clf__estimator__alpha': [0.1, 0.5, 1.0]  # Alpha parameter for Multinomial Naive Bayes
    }

    # Create the GridSearchCV object for Naive Bayes
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=5, scoring='accuracy')

    model = cv

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    return model, X_train, X_test, y_train, y_test



def evaluate_model(model, X_test, y_test, category_names):
    """
    Evaluate a machine learning model's performance on test data and print the results.

    This function takes a trained model, test features, test target categories, and a list
    of category names. It calculates and prints a classification report for each category
    and the overall model accuracy.

    Parameters:
        model: Trained machine learning model.
        X_test (pd.Series): Test features.
        y_test (pd.DataFrame): Test target categories.
        category_names (list): List of category names.

    Returns:
        Prints a classification report for each category and the overall model accuracy.

    """

    y_pred = model.predict(X_test)

    # Iterate through each column (feature)
    for i, col in enumerate(y_test.columns):
        print(f'Feature {i + 1}: {col}')
        
        # Calculate and print the classification report
        report = classification_report(y_test[col], y_pred[:, i])
        print(report)
        
    # Calculate overall accuracy
    accuracy = (y_pred == y_test.values).mean()
    print('The model accuracy is {:.3f}'.format(accuracy))



def save_model(model, model_filepath):
    """
    Save a trained machine learning model to a file.

    This function takes a trained model and a file path and saves the model to the specified file
    using pickle serialization.

    Parameters:
        model: Trained machine learning model to be saved.
        model_filepath (str): Filepath where the model will be saved.

    Returns:
        None
    """

    pickle.dump(model, open(model_filepath, "wb"))



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)  # Load data here
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model, X_train, X_test, y_train, y_test = build_model()  # Pass data to build_model
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()
