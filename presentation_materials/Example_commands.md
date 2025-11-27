# Use Venv
```sh
source venv/bin/activate
```

# Analyze code correctness using DeepSeek
```sh
python -m ai_feedback \
--submission_type jupyter \
--prompt code_explanation \
--scope code \
--submission presentation_materials/iris_image_examples/image_test_incorrect/student_submission.ipynb \
--question "4" \
--provider deepseek \
--model_name "deepSeek-R1:70b" \
--model_options max_tokens=5000

```

# Example Response
Let me identify some mistakes in your submission and explain why they occur:

1. **Line 25:** `df['species'] = iris.target`
   - **Why it's a mistake:** The `iris.target` array contains numerical values (0, 1, 2) representing species, but these numbers are not informative without context. This makes the data harder to interpret for someone unfamiliar with the dataset.
   - **Guidance:** Instead of directly assigning numerical values, you should map these values to their corresponding species names using `iris.target_names` first.

2. **Line 38:** `df['species name'] = iris.target_names[df['species']]`
   - **Why it's a mistake:** This line is correct in itself, but it would work better if the `species` column contained meaningful categorical values (like "setosa", "versicolor") instead of numerical values (0, 1, 2). Currently, it maps numbers to names, which works but isn't as intuitive.
   - **Guidance:** Consider modifying your earlier code to store species names directly in the `species` column and use this column for mapping.

3. **Lines 44-52:** `df.boxplot(...)`
   - **Why it's a mistake:** The `by='species'` parameter will create boxplots grouped by numerical values (0, 1, 2) because your `species` column contains numbers. This makes the plot less informative as the x-axis labels won't show the actual species names.
   - **Guidance:** Use the `species name` column instead of `species` for grouping to make the boxplots more interpretable.

# Analyze image correctness using OpenAI
```sh
python -m ai_feedback \
--submission_type jupyter \
--prompt image_analyze \
--scope image \
--submission_image presentation_materials/iris_image_examples/image_test_incorrect/student_submission.png \
--submission presentation_materials/iris_image_examples/image_test_incorrect/student_submission.ipynb \
--question "4" \
--provider openai

```

# Example Response
The graphs in the attached image do not fully solve the problem. While they do show side-by-side boxplots for sepal lengths, sepal widths, petal lengths, and petal widths, the x-axis labels use numeric codes (0, 1, 2) instead of the species names. The problem specifies that the ticks on the horizontal axes should be informative by using the species names.
