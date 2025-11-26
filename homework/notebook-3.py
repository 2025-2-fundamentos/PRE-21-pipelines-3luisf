#%%
def load_data():

    import pandas as pd

    dataframe = pd.read_csv(
        "../files/input/sentences.csv.zip",
        index_col=False,
        compression="zip",
    )

    data = dataframe.phrase
    target = dataframe.target

    return data, target
    
    
    
# %%
def make_train_test_split(x, y):

    from sklearn.model_selection import train_test_split

    (x_train, x_test, y_train, y_test) = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=123456,
    )
    return x_train, x_test, y_train, y_test
 
# %%
def make_pipeline(estimator):

    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from sklearn.pipeline import Pipeline

    vectorizer = CountVectorizer(
        lowercase=True,
        analyzer="word",
        token_pattern=r"\b[a-zA-Z]\w+\b",
        stop_words="english",
    )

    transformer = TfidfTransformer()

    pipeline = Pipeline(
        steps=[
            ("vectorizer", vectorizer),
            ("transformer", transformer),
            ("estimator", estimator),
        ],
        verbose=False,
    )

    return pipeline
# %%
def make_grid_search(estimator, param_grid, cv=5):

    from sklearn.model_selection import GridSearchCV

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring="balanced_accuracy",
    )

    return grid_search

# %%
def save_estimator(estimator):

    import pickle

    with open("estimator.pickle", "wb") as file:
        pickle.dump(estimator, file)
        
# %%
def load_estimator():

    import os
    import pickle

    if not os.path.exists("estimator.pickle"):
        return None
    with open("estimator.pickle", "rb") as file:
        estimator = pickle.load(file)

    return estimator
# %%
def train_logistic_regression():

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score

    data, target = load_data()

    x_train, x_test, y_train, y_test = make_train_test_split(
        x=data,
        y=target,
    )

    pipeline = make_pipeline(
        estimator=LogisticRegression(max_iter=1000),
    )

    estimator = make_grid_search(
        estimator=pipeline,
        param_grid={
            "transformer__norm": ["l1", "l2", None],
            "transformer__use_idf": [True, False],
            "transformer__smooth_idf": [True, False],
        },
        cv=5,
    )

    estimator.fit(x_train, y_train)

    best_estimator = load_estimator()

    if best_estimator is not None:

        saved_balanced_accuracy = balanced_accuracy_score(
            y_true=y_test, y_pred=best_estimator.predict(x_test)
        )

        current_balanced_accuracy = balanced_accuracy_score(
            y_true=y_test, y_pred=estimator.predict(x_test)
        )

        if current_balanced_accuracy < saved_balanced_accuracy:
            estimator = best_estimator

    save_estimator(estimator)


train_logistic_regression()

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
def check_estimator():

    import pickle

    import pandas as pd
    from sklearn.metrics import accuracy_score, balanced_accuracy_score

    data, target = load_data()

    x_train, x_test, y_train_true, y_test_true = make_train_test_split(
        x=data,
        y=target,
    )

    with open("estimator.pickle", "rb") as file:
        estimator = pickle.load(file)

    y_train_pred = estimator.predict(x_train)
    y_test_pred = estimator.predict(x_test)

    accuracy_train = round(accuracy_score(y_train_true, y_train_pred), 4)
    accuracy_test = round(accuracy_score(y_test_true, y_test_pred), 4)
    balanced_accuracy_train = round(
        balanced_accuracy_score(y_train_true, y_train_pred), 4
    )
    balanced_accuracy_test = round(balanced_accuracy_score(y_test_true, y_test_pred), 4)

    print(estimator.best_estimator_, ":", sep="")
    print(f"  Balanced Accuracy: {balanced_accuracy_test} ({balanced_accuracy_train})")
    print(f"           Accuracy: {accuracy_test} ({accuracy_train})")


check_estimator()

# %%
def train_mlp_classifier():

    from sklearn.metrics import balanced_accuracy_score
    from sklearn.neural_network import MLPClassifier

    data, target = load_data()

    x_train, x_test, y_train, y_test = make_train_test_split(
        x=data,
        y=target,
    )

    pipeline = make_pipeline(
        estimator=MLPClassifier(max_iter=10000),
    )

    estimator = make_grid_search(
        estimator=pipeline,
        param_grid={
            "transformer__norm": ["l1", "l2", None],
            "transformer__use_idf": [True, False],
            "transformer__smooth_idf": [True, False],
            "estimator__hidden_layer_sizes": [(1,), (5,), (5, 5)],
            "estimator__solver": ["adam"],
            "estimator__learning_rate_init": [0.01, 0.001, 0.0001],
        },
        cv=5,
    )

    estimator.fit(x_train, y_train)

    best_estimator = load_estimator()

    if best_estimator is not None:

        saved_balanced_accuracy = balanced_accuracy_score(
            y_true=y_test, y_pred=best_estimator.predict(x_test)
        )

        current_balanced_accuracy = balanced_accuracy_score(
            y_true=y_test, y_pred=estimator.predict(x_test)
        )

        if current_balanced_accuracy < saved_balanced_accuracy:
            estimator = best_estimator

    save_estimator(estimator)


train_mlp_classifier()
# %%
