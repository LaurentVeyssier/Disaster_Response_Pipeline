# Disaster Response Pipeline Project

![](./assets/wordcloud.png)

1. Project overview

This is the second project of [Udacity's Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025).
The project's objective is design a ML application supporting Disaster Emergency Response API. Tha app is composed of the following:
- front-end API where disaster messages can be submitted for classification inference
- back-end leveraging a trained classifier model for a Multi-label Classification task. The model tags the message along 36 different binary labels such as 'Food', 'Water', 'Medical support', 'Request (for)'
The app therefore allows to redirect messages to the appropriate first-line emergency response bodies.

2. Dataset used for model training

The datasets used for training the model has been provided by [Appen](https://www.figure-eight.com/) (formally Figure 8). The datasets are composed of :
- a message dataset collected from various sources during past disaster around the globe. It has a total of 26,248 text messages (original language and english translation)
- a categories dataset tagging each message along 36 binary labels (for classification).

Following elimination of duplicates and fixing binary variables performed during ETL, the final dataset retains 26,026 cleaned samples.

3. Architecture of the project
    - ETL pipeline: Loads the datasets, merges and clean data, stores in SQLite database
    - ML pipeline: Loads data from SQLite database. Segregate into train and test sets, Builds a text processing and machine learning pipeline, trains and fine-tunes a model using GridSearchCV, evaluates model performance and export final model as pickle file
    - Flask Web App: The flask app provides a classification inference API using the trained model

The project folder structure is as follows:

```
│   README.md
│   requirements.txt
│   ETL Pipeline Preparation.ipynb  # preparation notebook with data exploration
│   ML Pipeline Preparation.ipynb   # preparation notebook with training and feature engineering tests
│   categories.csv                  # training data to process
│   messages.csv                    # training data to process
│   LICENSE
│
├───app
│   │   run.py                       # script to run the flask API
│   │
│   └───templates
│           go.html                  # main page of web app
│           master.html              # classification result page of web app
│
├───assets                           # folder with figures for display
│
├───data
│       DisasterResponse.db          # sql database to store cleaned data
│       disaster_categories.csv      # data to process
│       disaster_messages.csv        # data to process
│       process_data.py              # script to clean and save datasets
│
└───models
        classifier.pk                # saved trained model
        model_performance.png
        train_classifier.py          # script to train the model
```

4. Binary label imbalance issue

The dataset suffers large imbalance issues:
- The top 5 labels representing over 60% of all positive labels (label=1). This means that the majority of labels have very few messages tagged, ie representing these labels.

![](./assets/label_imbalance.png)

- The number of positive labels is minimal for most of the labels with only a few exceptions. The label "related" shows the opposite situation with close to 80% positives. This means that circa 80% of the messages are tagged with "related" making it a "safe bet" for any classification model.

![](./assets/class_imbalance.png)

This is detrimental during training since a model will see a majority of these top samples and much less from those under-represented. This label imbalance issue is well described in this [article](https://medium.com/thecyphy/handling-data-imbalance-in-multi-label-classification-mlsmote-531155416b87).

The error rate on these under-represented labels is expected to be high. Trained model performance is therefore measured on each of the 36 labels separately using F1 score. To minimize this issue, data augmentation is performed on the labels with less than 1,000 available samples. Data augmentation is performed using NLPAug package which produces additional text samples by replacing words with synonyms. An illustration is shown below:

Original:
- `UN reports Leogane 80-90 destroyed. Only Hospital St. Croix functioning. Needs supplies desperately.`

Augmented Text:
- `united nations reports Leogane fourscore - 90 destroyed. But Infirmary St. Croix functioning. Needs supplies desperately.`
- `united nations account Leogane eighty - ninety destroyed. Solely Infirmary St. Croix functioning. Needs supplies desperately.`
- `UN reports Leogane eighty - ninety destroyed. Only Infirmary St. Croix operate. Needs provision desperately.`

5 augmentation steps are performed over the under-represented data, ie 5 new samples are created for each original sample. While the objective is to improve performance for low-represented labels, we should be careful not to introduce bias by oversampling a small set of messages. After data augmentation, our dataset increases to over 43,600 samples.

An alternative to data augmentation would be to use a reduce dataset with equal proportion of labels. However since some labels have so few samples, this would imply reducing the dataset a lot resulting in the loss of massive training information.


5. Training step:

During preparation, XGBoost demonstrated higher performance compared to ramdomforest. GridSearchCV was performed but with minimal improvement over standard parameters. Several feature engineering approaches were tested with minimal performance improvements:
- length of the message
- use of `genre` categorical variable. each message has this information. There are 3 sources: 'news', 'social' or 'direct'. I would have expected this could add classification information but the data analysis performed during ETL did not show any particular correlation with specific labels.
- messages starting with a verb

The data augmentation improved performance significantly on the under-represented binary labels.

The training pipeline was therefore composed of:
- data augmentation on less represented labels
- tokenization and vectorization using tfIdf. parameters were set at min 3 occurences and lax 10,000 features to prevent memory issues and reduce training time without noticeable performance deterioration
- GridSearch hyperparameter tuning (n_estimators, max_depth)
- Training for Multi-label Classification task. In this type of classification problem the target variable has more than one dimension where each dimension is binary i.e. contain only two distinct values (0 or 1).

Below is the overview of model performance in the various testing conditions. Saved model is XGBoost with 5 data augmentation steps (last column to the right). The model achieves **above 80% unweighted F1 score on nearly all labels**.

![](./assets/fine-tuning.png)


6. Front-end API

The front-end displays some insights extratced from the dataset.
    - wordcloud using most frequent words in the messages dataset. I used my tokenization step from production.
    - Label imbalance overview
    - message inference (illustration below)

![](./assets/message_inference.png)


7. Environment set-up

The project runs in a virtual environment using python 3.8 and the python packages provided in the requirements.txt file


8. Instructions
    1. Run the following commands in the project's root directory to set up your database and model.

        - To run ETL pipeline that cleans data and stores in database
            `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        - To run ML pipeline that trains classifier and saves
            `python models/train_classifier.py data/DisasterResponse.db models/classifier.pk`

    2. Go to `app` directory: `cd app`

    3. Run your web app: `python run.py` 

    4. Click to open the browser at the API specified address

![](./assets/front_end.png)

![](./assets/message_inference2.png)
