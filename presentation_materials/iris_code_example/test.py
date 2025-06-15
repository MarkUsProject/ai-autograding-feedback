import numpy as np
import pandas as pd
import pytest
from instructor_solution import (
    calculate_statistics as instructor_calculate_statistics,  # Instructor's implementation
)
from instructor_solution import classify_single_species as instructor_classify_species

# Load the Iris dataset
from sklearn.datasets import load_iris
from student_submission import (  # Student's implementation
    calculate_statistics,
    classify_single_species,
)

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Create a binary target variable by selecting only two species (Setosa and Versicolor)
df["species"] = iris.target
df = df[df["species"] != 2]  # Remove the Virginica species (category 2)
df["species"] = df["species"].apply(lambda x: 1 if x == 1 else 0)  # Setosa = 0, Versicolor = 1


# Test sepal_length_mean
def test_sepal_length_mean():
    student_statistics = calculate_statistics(df)
    instructor_statistics = instructor_calculate_statistics(df)
    assert np.isclose(
        student_statistics["sepal_length_mean"],
        instructor_statistics["sepal_length_mean"],
        atol=0.1,
    )


# Test sepal_width_mean
def test_sepal_width_mean():
    student_statistics = calculate_statistics(df)
    instructor_statistics = instructor_calculate_statistics(df)
    assert np.isclose(
        student_statistics["sepal_width_mean"],
        instructor_statistics["sepal_width_mean"],
        atol=0.1,
    )


# Test sepal_length_std
def test_sepal_length_std():
    student_statistics = calculate_statistics(df)
    instructor_statistics = instructor_calculate_statistics(df)
    assert np.isclose(
        student_statistics["sepal_length_std"],
        instructor_statistics["sepal_length_std"],
        atol=0.1,
    )


# Test sepal_width_std
def test_sepal_width_std():
    student_statistics = calculate_statistics(df)
    instructor_statistics = instructor_calculate_statistics(df)
    assert np.isclose(
        student_statistics["sepal_width_std"],
        instructor_statistics["sepal_width_std"],
        atol=0.1,
    )


# Test sepal_length_median
def test_sepal_length_median():
    student_statistics = calculate_statistics(df)
    instructor_statistics = instructor_calculate_statistics(df)
    assert np.isclose(
        student_statistics["sepal_length_median"],
        instructor_statistics["sepal_length_median"],
        atol=0.1,
    )


# Test sepal_width_median
def test_sepal_width_median():
    student_statistics = calculate_statistics(df)
    instructor_statistics = instructor_calculate_statistics(df)
    assert np.isclose(
        student_statistics["sepal_width_median"],
        instructor_statistics["sepal_width_median"],
        atol=0.1,
    )


# Test sepal_length_var
def test_sepal_length_var():
    student_statistics = calculate_statistics(df)
    instructor_statistics = instructor_calculate_statistics(df)
    assert np.isclose(
        student_statistics["sepal_length_var"],
        instructor_statistics["sepal_length_var"],
        atol=0.1,
    )


# Test sepal_width_var
def test_sepal_width_var():
    student_statistics = calculate_statistics(df)
    instructor_statistics = instructor_calculate_statistics(df)
    assert np.isclose(
        student_statistics["sepal_width_var"],
        instructor_statistics["sepal_width_var"],
        atol=0.1,
    )


# Test predictions for a single data point (edge case for individual prediction)
def test_single_prediction_1():
    single_sample = df.iloc[0:1, :-1]  # Select just the features (excluding the species label)

    # Student's and instructor's predictions for the single sample
    student_single_prediction = classify_single_species(single_sample)
    instructor_single_prediction = instructor_classify_species(single_sample)

    # Verify both predictions are the same for the single data point
    assert student_single_prediction == instructor_single_prediction


def test_single_prediction_2():
    single_sample = df.iloc[5:6, :-1]  # Select just the features (excluding the species label)

    # Student's and instructor's predictions for the single sample
    student_single_prediction = classify_single_species(single_sample)
    instructor_single_prediction = instructor_classify_species(single_sample)

    # Verify both predictions are the same for the single data point
    assert student_single_prediction == instructor_single_prediction


def test_single_prediction_3():
    single_sample = df.iloc[6:7, :-1]  # Select just the features (excluding the species label)

    # Student's and instructor's predictions for the single sample
    student_single_prediction = classify_single_species(single_sample)
    instructor_single_prediction = instructor_classify_species(single_sample)

    # Verify both predictions are the same for the single data point
    assert student_single_prediction == instructor_single_prediction
