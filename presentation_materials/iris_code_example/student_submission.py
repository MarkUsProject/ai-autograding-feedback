import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Create a binary target variable by selecting only two species (Setosa and Versicolor)
df["species"] = iris.target
df = df[df["species"] != 2]  # Remove the Virginica species (category 2)
df["species"] = df["species"].apply(lambda x: 1 if x == 1 else 0)  # Setosa = 0, Versicolor = 1


def calculate_statistics(df):
    ### TO DO ###
    sepal_length = df["sepal length (cm)"]
    sepal_width = df["sepal width (cm)"]

    statistics = {
        "sepal_length_mean": np.mean(sepal_length),
        "sepal_length_median": np.median(sepal_length),
        "sepal_length_std": np.std(sepal_length),
        "sepal_length_var": np.var(sepal_length),
        "sepal_width_mean": np.mean(sepal_length),
        "sepal_width_median": np.mean(sepal_width),
        "sepal_width_std": np.std(sepal_width),
        "sepal_widdth_var": np.var(sepal_width),
    }
    ################

    return statistics


X = df[["sepal length (cm)", "sepal width (cm)"]]
y = df["species"]
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)
### TO DO ###
model = LinearRegression()
################
model.fit(X_train, y_train)


def classify_single_species(sample):
    ### TO DO ###
    X = sample[["sepal length (cm)", "sepal width (cm)"]]
    prediction = model.predict(X)[0]
    prediction = round(prediction)
    ################
    return prediction
