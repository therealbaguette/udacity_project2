import nltk
nltk.download(['punkt', 'wordnet'])

import sys

import re
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import pickle


def load_data(database_filepath):

    """
    Load data from a SQLite database.

    Args:
    database_filepath (str): The filepath of the SQLite database.

    Returns:
    tuple: A tuple containing X (features), Y (labels), and category_names.
    """

    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('disaster_response', con=engine)
    X = df['message']
    Y = df[df.columns[4:]]
    category_names = Y.columns
      
    return X, Y, category_names


def tokenize(text):

    """
    Tokenize text data.

    Args:
    text (str): The text to be tokenized.

    Returns:
    list: A list of cleaned tokens.    """

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    detected_urls = re.findall(url_regex, text)
    
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():

    """
    Build a machine learning pipeline.

    Returns:
    GridSearchCV: A GridSearchCV object representing the machine learning pipeline.
    """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])


    # smaller set of parameters to reduce runtime and pickle file size
    parameters = {
    'vect__max_df': [0.5, 0.75],  # Maximum document frequency for CountVectorizer
    'tfidf__use_idf': [True, False]  # Whether to use IDF in TfidfTransformer
    }

    # parameters = {
    # 'vect__max_df': [0.5, 0.75, 1.0],  # Maximum document frequency for CountVectorizer
    # 'tfidf__use_idf': [True, False],  # Whether to use IDF in TfidfTransformer
    # 'clf__estimator__n_estimators': [50, 100, 200],  # Number of trees in the random forest
    # 'clf__estimator__max_depth': [None, 10, 20]  # Maximum depth of the trees in the random forest
    # }


    model_opt = GridSearchCV(pipeline, param_grid=parameters, cv=3, verbose=2, n_jobs=-1)

    # cv.fit(X_train, y_train)

    # model_opt = cv.best_estimator_

    return model_opt


def evaluate_model(model, X_test, Y_test, category_names):

    """
    Evaluate the performance of the model.

    Args:
    model (GridSearchCV): The trained model.
    X_test (DataFrame): Test features.
    Y_test (DataFrame): Test labels.
    category_names (list): List of category names.

    Returns:
    None
    """

    Y_pred = model.predict(X_test)

    # class_report = classification_report(Y_test, Y_pred, target_names=category_names)

    # class_report = classification_report(Y_test, Y_pred)

    for i in range(36):
        print(Y_test.columns[i], ':')
        print(classification_report(Y_test.iloc[:,i], Y_pred[:,i]))
    
    # print(class_report)



def save_model(model, model_filepath):

    """
    Save the model to a file.

    Args:
    model (GridSearchCV): The trained model.
    model_filepath (str): The filepath where the model will be saved.

    Returns:
    None
    """

    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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

    # python models/train_classifier.py ../data/DisasterResponse.db classifier.pkl

    # python train_classifier.py data/DisasterResponse.db classifier.pkl