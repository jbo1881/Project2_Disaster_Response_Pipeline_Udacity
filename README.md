# Project 2: Disaster Response Pipeline (Udacity)

## Table of Contents

- [Installation](#installation)
- [Project Motivation](#project-motivation)
- [Project Description](#project-description)
- [Files Description](#files-description)
- [Instructions](#instructions)
- [Results](#results)
- [Licensing, Authors, Acknowledgements](#licensing-authors-acknowledgements)

## Installation

Before running the project, ensure you have the following libraries installed:

- **pandas**: This library is essential for data manipulation and analysis. It offers versatile data structures like DataFrames and Series, simplifying tabular data handling.

- **numpy**: NumPy is a fundamental package for scientific computing with Python. It provides support for arrays, matrices, and mathematical functions, making complex computations efficient.

- **sqlite3**: Python's built-in module for SQLite database management. It allows you to create, connect to, and manipulate SQLite databases without requiring external installations.

- **sqlalchemy**: SQLAlchemy is a widely-used SQL toolkit and Object-Relational Mapping (ORM) library for Python. It abstracts SQL database interactions and simplifies database access.

- **sys**: The sys module provides access to variables used or maintained by the Python interpreter. In this project, it's used for accessing command-line arguments.

- **re**: The re module is employed for working with regular expressions, powerful tools for pattern matching and text manipulation.

- **nltk**: The Natural Language Toolkit is a versatile library for processing human language data. It offers interfaces to numerous corpora and linguistic resources, such as WordNet, making it invaluable for text and natural language processing (NLP) tasks.

- **sklearn**: Scikit-learn is a comprehensive machine learning library for Python. It simplifies data mining and analysis with efficient tools. In this code, it's used to create a machine learning model pipeline, perform data splitting, and assess model performance.

- **pickle**: Python's pickle module is utilized for serializing and deserializing Python objects. It plays a crucial role in saving and loading machine learning models.


## Project Motivation

The project's objective is the categorization of disaster messages. To achieve this, I examined disaster data sourced from [Appen](https://appen.com/) with the aim of constructing an API model for message classification. Users can utilize a web application to input new messages and receive classifications across various categories. Additionally, the web app provides data visualizations for enhanced insights.

## Project Components

The project consists of the following components:

### 1. ETL Pipeline

In the `process_data.py` Python script, an ETL (Extract, Transform, Load) pipeline is implemented, which performs the following tasks:

- Loads the messages and categories datasets.
- Merges the two datasets.
- Cleans the data by transforming it into a structured format.
- Stores the cleaned data in a SQLite database.

### 2. ML Pipeline

The `train_classifier.py` Python script contains a machine learning pipeline that carries out the following steps:

- Loads data from the SQLite database generated by the ETL pipeline.
- Splits the dataset into training and test sets.
- Constructs a text processing and machine learning pipeline.
- Trains and fine-tunes a machine learning model using GridSearchCV.
- Evaluates and reports the model's performance on the test set.
- Exports the final model as a pickle file for future use.

### 3. Flask Web App

A web application is provided, allowing users to interact with the trained model. Key features of the web app include:

- Inputting a disaster message.
- Viewing the categories associated with the message.

 
## Files Description

The file structure is organized as follows:

	- README.md: read me file
 	- [GitHub Repository](https://github.com/jbo1881/Project2_Disaster_Response_Pipeline_Udacity/tree/main)
	- ETL Pipeline Preparation.ipynb: contains ETL pipeline preparation code
	- ML Pipeline Preparation.ipynb: contains ML pipeline preparation code
	- \app
			- run.py: flask file to run the app
		- \templates
			- master.html: main page of the web application 
			- go.html: result web page
	- \data
			- disaster_categories.csv: categories dataset
			- disaster_messages.csv: messages dataset
			- Disaster_Messages.db: disaster messages database
			- process_data.py: ETL process
	- \models
			- train_classifier.py: ML classification code


## Instructions

To run the application, follow the provided guidelines:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/Disaster_Messages.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/Disaster_Messages.db`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

## Results

1. **ETL Pipeline**: ETL (Extract, Transform, Load) pipeline to read data from two CSV files, perform data cleaning operations, and subsequently store the processed data in an SQLite database.

2. **Machine Learning Pipeline**: The project includes a machine learning pipeline designed to train a classifier capable of conducting multi-output classification across the 36 categories present in the dataset.

3. **Flask Web Application**: Flask-based web application that not only showcases data visualizations but also offers message classification functionality. Users can input messages through the web interface to get category predictions.

## Licensing, Authors, Acknowledgements

Acknowledgments go to Udacity for providing the initial codebase and Appen for supplying the dataset utilized in this project.

##








































