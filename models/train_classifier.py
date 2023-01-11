import nlpaug.augmenter.word as naw
import sys
import re
import pickle
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

# for modeling tasks
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

# for NLP tasks
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

# NLP data augmentation
aug = naw.SynonymAug(aug_src='wordnet')


def load_data(database_filepath, augmentation=True):
    """read sql table from database path and return segregated content
    inputs:
        - database_filepath: path to database (str)
    outputs:
        - X: dataframe with messages as str
        - Y: labels (dataframe)
        - category_names: list of classification labels (list of str)
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table(table_name='messages', con=engine)

    if augmentation:
        df = data_augmentation(df)

    # df.select_dtypes(include=['object']).drop('original',axis=1)
    X = df.message
    # There are 36 classes
    Y = df.select_dtypes(include=['int64'])
    category_names = Y.columns.tolist()
    return X, Y, category_names


def data_augmentation(dataframe):
    # select classification labels with less than 1000 samples
    labels = dataframe.select_dtypes(include=['int64']).columns.tolist()
    sorted_labels = dataframe[labels].sum(axis=0).sort_values().index
    under_represented_labels = sorted_labels[dataframe[labels].sum(
        axis=0).sort_values() < 1000]

    # perform data augmentation on under-represented classes
    new_rows = []
    df_to_augment = dataframe.loc[dataframe[under_represented_labels].sum(
        axis=1) == 1, :]
    for idx, row in df_to_augment.iterrows():
        sample = row['message']
        augmented_text = aug.augment(sample)
        new_row = row.copy()
        new_row['message'] = augmented_text[0]
        new_rows.append(new_row)

    # combine new augmented rows with existing dataset
    augmented_df = pd.DataFrame(new_rows)
    augmented_df = pd.concat([dataframe, augmented_df], axis=0)
    augmented_df.reset_index(drop=True, inplace=True)
    return augmented_df


def tokenize(text):
    """performs preprocessing on a string and return takenized text
        - normalize text to lowercase
        - strip all characters not a digit or a letter
        - tokenize to words
        - remove stopwords
        - lemmatize nouns and verbs to root
    inputs:
        text: str
    outputs:
        tokens: list of tokens
    """
    # normalize case and remove punctuation
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(token)
              for token in tokens if token not in stop_words]
    tokens = [lemmatizer.lemmatize(token, pos='v') for token in tokens]

    return tokens


def build_model():
    """Build ML pipeline for model training
    inputs:
        - None
    outputs:
        - ML model
    """
    pipeline = Pipeline([('tokenization',
                          CountVectorizer(tokenizer=tokenize,
                                          min_df=3,
                                          max_features=10000)),
                         ('vectorization',
                          TfidfTransformer()),
                         ('modelisation',
                          MultiOutputClassifier(estimator=XGBClassifier(),
                                                n_jobs=-1))])
    return pipeline


def evaluate_model(model, x, y, category_names, display_perf=False):
    """Evaluate classification model performance
    inputs
        - model: trained ML model
        - x: X_test features (dataframe)
        - y: target label corresponding to X_text features (dataframe)
        - category_names: list of str with classification labels

    """
    yhat = model.predict(x)
    labels = y.columns.tolist()
    micro_score = []
    for i, name in enumerate(category_names):
        print(f'******************** class: {labels[i]} *********************')
        cr = classification_report(y.iloc[:, i], yhat[:, i], zero_division=0)
        micro_score.append(round(f1_score(y.iloc[:, i],
                                          yhat[:, i],
                                          average='macro'), 2))
        print(cr)
        print()
    perf = pd.DataFrame(micro_score, index=labels, columns=[name])
    perf.plot(kind='bar', figsize=(15, 5), label='XGBoostClassifier')
    plt.title("Macro F1 score per label (unweighted)")
    plt.tight_layout()
    plt.savefig('model_performance.png')
    if display_perf:
        plt.show()


def save_model(model, model_filepath):
    """save trained model to pickle file at provided filepath"""
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """ pipeline to load dataset, segregate train / test data,
        assemble, train and savel ML model
        data augmentation performed by default using nlpaug package
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                            random_state=10,
                                                            test_size=0.2)

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
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pk')


if __name__ == '__main__':
    main()
