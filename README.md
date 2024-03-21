# Disaster Response Pipeline Project

## Table of Contents
1. [Overview](#overview)
2. [File Structure](#file-structure)
3. [Instructions](#instructions)

## <a name="overview"></a>Overview

This project is part of the Udacity Data Scientist Nanodegree Program, created in collaboration with Appen. Its main objective is to develop a model capable of categorizing disaster-related messages received during a crisis, ensuring effective and timely communication between those in need and relevant response agencies. The project culminates in a web application where users can input messages for classification.

## <a name="file-structure"></a>File Structure

### app
- **run.py**: Python script to launch the web application.
- **templates**: Folder containing web application templates (go.html & master.html).

### data
- **disaster_messages.csv**: Dataset containing real messages sent during disaster events (provided by Figure Eight).
- **disaster_categories.csv**: Dataset containing categories of the messages.
- **process_data.py**: ETL (Extract, Transform, Load) pipeline used to clean, preprocess, and store data in a SQLite database.
- **ETL Pipeline Preparation.ipynb**: Jupyter Notebook used for ETL pipeline development.
- **DisasterResponse.db**: SQLite database containing cleaned data.

### models
- **train_classifier.py**: Machine learning pipeline used to load cleaned data, train the model, and save the trained model as a pickle (.pkl) file.
- **classifier.pkl**: Pickle file containing the trained model.
- **ML Pipeline Preparation.ipynb**: Jupyter Notebook used for machine learning pipeline development.


## <a name="instructions"></a>Instructions

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage