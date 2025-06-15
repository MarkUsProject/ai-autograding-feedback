"""Automated tests for GGR274 Homework 5

Installation instructions (if running locally)
==============================================

This test script requires a few different libraries, which you can install with:

$ pip install pytest
$ pip install 'git+https://github.com/MarkUsProject/autotest-helpers.git#subdirectory=notebook_helper'
$ pip install 'git+https://github.com/UofTCompDSci/cds-testing'

After installing the libraries, you can run the tests in this file from the command line using:

$ pytest test_hw5.py
"""

# This is a library for helping write some tests
import cds_testing
import Homework_5_solution as solution
import numpy as np
import pandas as pd
import pytest

# This imports the student notebook, but does not yet run all the code cells
# The module name needs to matche the student file name, omitting the .ipynb extension
import test1_submission as hw

# This import is required to import Jupyter notebooks like regular Python files
from notebook_helper import importer


@pytest.fixture(scope="module", autouse=True)
def run_notebook():
    """This executes the code in the notebook before running the tests."""
    importer.run_cells(hw, raise_on_error=False)


# This executes the code in the student and solution notebooks before running the tests.
load_student_code = pytest.fixture(cds_testing.load_code, params=[hw], scope="module", autouse=True)
load_solution_code = pytest.fixture(cds_testing.load_code, params=[solution], scope="module", autouse=True)


def test_time_boxplots():
    """Test that the 'time_boxplots' variable has been defined
    correctly in your notebook.
    """

    # Check that the variable exists in the notebook
    assert hasattr(hw, "time_boxplots"), 'We could not find a variable called "time_boxplots" in your file.'

    # Make sure that the variable is string
    assert isinstance(hw.time_boxplots, np.ndarray), "Your time_boxplots variable is incorrectly defined."

    flattened_boxplot_hw = hw.time_boxplots.flatten()
    flattened_boxplot_soln = solution.time_boxplots.flatten()

    assert (
        len(flattened_boxplot_hw) >= 3
    ), 'The figure in the variable "time_boxplots" does not have the correct number of boxplots '

    # Depending on the layout of the boxplots, there may be some axes that are empty.
    # For example if you have a 3 boxplots on a 2x2 grid, one of the titles will be an empty string
    # so we filter it when looping through all the titles and labels.
    hw_titles = [ax.get_title() for ax in flattened_boxplot_hw if len(ax.get_title()) > 0]
    soln_titles = [ax.get_title() for ax in flattened_boxplot_soln if len(ax.get_title()) > 0]

    hw_labels = [ax.get_xlabel() for ax in flattened_boxplot_hw if len(ax.get_xlabel()) > 0]
    soln_labels = [ax.get_xlabel() for ax in flattened_boxplot_soln if len(ax.get_xlabel()) > 0]

    assert sorted(hw_titles) == sorted(
        soln_titles
    ), 'The boxplots in "time_boxplots" should have the following axis titles: "Socializing time (hour)", \
        "Exercising time (hour)", "Sleep time (hour)" (in any order)'

    assert sorted(hw_labels) == sorted(
        soln_labels
    ), 'The boxplots in "time_boxplots" should be created by grouping using the Age group variable'


def test_time_boxplots_age_label():
    """Test that the 'time_boxplots_age_label' variable has been defined
    correctly in your notebook.
    """

    # Check that the variable exists in the notebook
    assert hasattr(
        hw, "time_boxplots_age_label"
    ), 'We could not find a variable called "time_boxplots_age_label" in your file.'

    # Make sure that the variable is string
    assert isinstance(
        hw.time_boxplots_age_label, np.ndarray
    ), "Your time_boxplots_age_label variable is incorrectly defined."

    flattened_boxplot_hw = hw.time_boxplots_age_label.flatten()
    flattened_boxplot_soln = solution.time_boxplots_age_label.flatten()

    assert (
        len(flattened_boxplot_hw) >= 3
    ), 'The figure in the variable "time_boxplots_age_label" does not have the correct number of boxplots '

    # Depending on the layout of the boxplots, there may be some axes that are empty.
    # For example if you have a 3 boxplots on a 2x2 grid, one of the titles will be an empty string
    # so we filter it when looping through all the titles and labels.
    hw_titles = [ax.get_title() for ax in flattened_boxplot_hw if len(ax.get_title()) > 0]
    soln_titles = [ax.get_title() for ax in flattened_boxplot_soln if len(ax.get_title()) > 0]

    hw_labels = [ax.get_xlabel() for ax in flattened_boxplot_hw if len(ax.get_xlabel()) > 0]
    soln_labels = [ax.get_xlabel() for ax in flattened_boxplot_soln if len(ax.get_xlabel()) > 0]

    assert sorted(hw_titles) == sorted(
        soln_titles
    ), 'The boxplots in "time_boxplots_age_label" should have the following axis titles: "Socializing time (hour)", \
        "Exercising time (hour)", "Sleep time (hour)" (in any order)'

    assert sorted(hw_labels) == sorted(
        soln_labels
    ), 'The boxplots in "time_boxplots_age_label" should be created by grouping using the Age group label variable'


def test_time_use_subset_renamed_df_task1c():
    """Test that the 'time_use_subset_renamed_df' dataframe is correctly created in your notebook up to Task 1c."""
    # Check that the variable exists in the notebook
    assert hasattr(
        hw, "time_use_subset_renamed_df"
    ), 'We could not find a variable called "time_use_subset_renamed_df" in your file.'

    # Make sure that the variable is a pandas dataframe
    assert isinstance(
        hw.time_use_subset_renamed_df, pd.DataFrame
    ), "Your time_use_subset_renamed_df variable is incorrectly defined. It should refer to a Pandas dataframe."

    # Copy the columns required only for task 1c.
    hw_df_copy = hw.time_use_subset_renamed_df[
        ["Socializing time", "Exercising time", "Sleep time", "Age group", "Province"]
    ]
    soln_df_copy = solution.time_use_subset_renamed_df[
        ["Socializing time", "Exercising time", "Sleep time", "Age group", "Province"]
    ]

    pd.testing.assert_frame_equal(hw_df_copy, soln_df_copy, obj="(Task 1c) time_use_subset_renamed_df")


# Check for the exact values of the variables
TASK_VARS = {
    "time_use_df": {},  # Task 1a
    "analysis_columns": {},  # Task 1b
    "time_use_subset_df": {},  # Task 1b
    "new_col_names": {},  # Task 1c
    "time_use_subset_renamed_df": {},  # Task 1c
    "time_use_subset_renamed_df": {},  # Task 2
    "well_balanced": {},  # Task 3a
    "well_balanced_df": {},  # Task 3b
    "diff": {},  # Task 3c
    "pct_lost": {},  # Task 3d
    "group_means": {},  # Task 4acd
    "group_means_sorted": {},  # Task 4e
    "well_balanced_age_label_df": {},  # Task 5a
}

test_variable_names = cds_testing.make_variable_names_test(hw, TASK_VARS)
test_expected_value = cds_testing.make_answer_equality_test(hw, solution, TASK_VARS)

if __name__ == "__main__":
    pytest.main(["--json=results.json", "test_hw5.py"])
