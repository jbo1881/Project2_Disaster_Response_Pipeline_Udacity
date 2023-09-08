# Project2_Disaster_Response_Pipeline_Udacity

## Table of Contents

- [Installation](#installation)
- [Project Motivation](#project-motivation)
- [Project Description](#project-description)
- [Files Description](#files-description)

## Installation

The libraries used are: 

  pandas: This library is used for data manipulation and analysis. It provides data structures like DataFrames and Series, making it easier to work with tabular data.
  
  numpy: NumPy is a fundamental package for scientific computing with Python. It provides support for arrays, matrices, and mathematical functions.
  
  sqlite3: This library is Python's built-in module for working with SQLite databases. It allows you to create, connect to, and interact with SQLite databases without the need for external installations.
  
  sqlalchemy: SQLAlchemy is a popular SQL toolkit and Object-Relational Mapping (ORM) library for Python. It's used for working with SQL databases more abstractly.
  
  sys: This module provides access to some variables used or maintained by the interpreter and to functions that interact with the interpreter. It is used to access command-line arguments.
  
  re: The re module is used for regular expressions, which are powerful tools for pattern matching and text processing.
  
  nltk: The Natural Language Toolkit is a library for working with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources, such as WordNet. It's used for text processing and natural language processing (NLP) tasks.
  
  sklearn: Scikit-learn is a machine learning library for Python. It provides simple and efficient tools for data mining and data analysis. In the code, it is being used to create a machine learning model pipeline, perform data splitting, and evaluate the model's performance.
  
  pickle: Python's pickle module is used for serializing and deserializing Python objects. In the code, it's used to save and load machine learning models.



## Project Motivation

The project's objective is the categorization of disaster messages. To achieve this, I examined disaster data sourced from Appen with the aim of constructing an API model for message classification. Users can utilize a web application to input new messages and receive classifications across various categories. Additionally, the web app provides data visualizations for enhanced insights.

## Project Description

The parts of the project are as follows:

1. ETL Pipeline: In a Python script, process_data.py, a data cleaning pipeline performs:

- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

  2. ML Pipeline: In a Python script, train_classifier.py, a machine learning pipeline executes:

- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

  3. Flask Web App: The web application allows users to input a disaster message and subsequently observe the message's associated categories.
 
  ## Files Description

