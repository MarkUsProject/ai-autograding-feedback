# Use Venv
```sh
source ai-autograding-feedback/LlmVenv/bin/activate
```

# Analyze code correctness using Claude
```sh
python -m ai_feedback \
--submission_type jupyter \
--prompt code_explanation \
--scope code \
--assignment presentation_materials/iris_image_examples/image_test_incorrect \
--question "4" \
--model claude-3.7-sonnet \
--output stdout
```

# Example Response
## Feedback for Task 4

## Mistake in part a:
```python
df['species name'] = iris.target_names[df['species']]
```

This line contains an error because you're trying to use a Series (`df['species']`) as an index for the `iris.target_names` array. You can't directly index a NumPy array with a Series like this. When you try to use a Series to index into an array, Python expects each element of the Series to be a valid index for the array, but it tries to do this all at once rather than element by element.

## Mistake in part b:
```python
boxplots_labelled = df.boxplot(
    column=[
        'sepal length (cm)',
        'sepal width (cm)',
        'petal length (cm)',
        'petal width (cm)'],
    by='species',
    figsize=(10, 10)
)
```

There are two issues with this code:
1. You're using `by='species'` when the instructions specifically ask you to use the `'species name'` column for the boxplot grouping
2. You're using `df` instead of `df_labelled` as specified in the instructions

The boxplot should be grouped by the more informative `'species name'` column so that the x-axis labels show the actual species names rather than numeric codes.

# Analyze image correctness using OpenAI
```sh
python -m ai_feedback \
--submission_type jupyter \
--prompt image_analyze \
--scope image \
--assignment presentation_materials/iris_image_examples/image_test_incorrect \
--question "4" \
--model openai \
--output stdout
```

# Example Response
The graphs in the image do not fully solve the problem as specified. While they do show side-by-side boxplots for sepal lengths, sepal widths, petal lengths, and petal widths, the horizontal axis uses numerical labels (0, 1, 2) instead of the species names. The problem asks for the species names to be used, making the ticks on the horizontal axes informative.
