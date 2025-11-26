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
    
#%%

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
def save_estimator(estimator):

    import pickle

    with open("estimator.pickle", "wb") as file:
        pickle.dump(estimator, file)
        
        
        
# %%   
def train_logistic_regression():

    from sklearn.linear_model import LogisticRegression

    data, target = load_data()
    x_train, x_test, y_train, y_test = make_train_test_split(
        x=data,
        y=target,
    )
    estimator = make_pipeline(estimator=LogisticRegression(max_iter=1000))
    estimator.fit(x_train, y_train)
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