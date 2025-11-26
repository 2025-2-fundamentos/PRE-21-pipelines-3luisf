#%%
def train_estimator():

    import pandas as pd
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    # import os
    import pickle

    #
    # Manejo de la data
    #
    dataframe = pd.read_csv(
        "../files/input/sentences.csv.zip",
        index_col=False,
        compression="zip",
    )

    data = dataframe.phrase
    target = dataframe.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        data,
        target,
        test_size=0.3,
        shuffle=False,
    )

    #
    # Modelo de regresión logística
    #
    vectorizer = CountVectorizer(
        lowercase=True,
        analyzer="word",
        token_pattern=r"\b[a-zA-Z]\w+\b",
        stop_words="english",
    )

    transformer = TfidfTransformer()

    lr_estimator = Pipeline(
        steps=[
            ("vectorizer", vectorizer),
            ("transformer", transformer),
            ("estimator", LogisticRegression(max_iter=1000)),
        ],
        verbose=False,
    )

    lr_estimator.fit(X_train, y_train)

    with open("estimator.pickle", "wb") as file:
        pickle.dump(lr_estimator, file)

train_estimator()


# %%
def use_estimator():

    import pickle

    import pandas as pd

    dataframe = pd.read_csv(
        "../files/input/sentences.csv.zip",
        index_col=False,
        compression="zip",
    )

    data = dataframe.phrase

    with open("estimator.pickle", "rb") as file:
        estimator = pickle.load(file)

    prediction = estimator.predict(data)

    return prediction


use_estimator()


# %%
