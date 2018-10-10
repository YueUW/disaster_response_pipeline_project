# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.externals import joblib

import nltk
nltk.download(['punkt', 'wordnet'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    # load data from database
    # argument: database file path
    # return: X: feature variables
    #         Y: target variables
    #         col_names: column names
    
    # load data from database to dataframe df
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponseTable', con=engine) 
    
    # extract feature variables, target variables and column names for target variables
    X = df.message.values
    col_names = df.columns
    Y = df[col_names[4:]].values

    return X, Y, col_names

def tokenize(text):
    # apply tokenization function to text variable
    # argument: text variables
    # return: cleaned variable after tokenization
    
    # initialize tokens and lemmatizer
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    # apply tokenization to each token
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    # build machine learning pipeline and apply grid search
    # input: none
    # return: machine learning model
    
    # build pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # parameters for grid search
    # some paramters are commented out to save running time
    parameters = {
    #'vect__ngram_range': ((1, 1), (1, 2)),
    #'vect__max_df': (0.5, 0.75, 1.0),
    #'vect__max_features': (None, 5000, 10000),
    'tfidf__use_idf': (True, False),
    #'clf__estimator__n_estimators': [10, 100],
    #'clf__estimator__min_samples_split': [2, 3, 4],
    #'clf__estimator__max_features': ['auto', 'log2', None]
    }
    
    # apply grid search on the pipeline
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    # predict test set and compare with real test set for model evaluation
    # input: machine learning model;
    #       test set feature variables
    #       test set target variables
    #       category name for target variables
    # return: none
    
    # print out best parameters that grid search finds
    print("Best parameters found by grid search:")
    print(model.best_params_)
    
    # predict test set
    Y_pred = model.predict(X_test)
    
    # print out classification report and confusion matrix for each individual category 
    print("Classification report:")
    for i in range(Y_pred.shape[1]):
        print('Column ',i, ': ', category_names[4+i], '\n', classification_report(Y_test[:,i], Y_pred[:,i]))
        print(confusion_matrix(Y_test[:,i], Y_pred[:,i]))


def save_model(model, model_filepath):
    # save model to certain file path
    # input: trained machine learning model
    #       target file path
    # return: none
    
    # save model
    joblib.dump(model, model_filepath)


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