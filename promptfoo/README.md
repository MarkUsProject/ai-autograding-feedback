## Developer Guide: Using Promptfoo for Prompt Evaluation

This section explains how to run and view prompt evaluations using [Promptfoo](https://promptfoo.dev/) in the `ai-autograder-feedback` project.

---

### Running Tests

1. **SSH into the Polymouth server.**
2. **Clone** the `ai-autograder-feedback` repository (if you haven't already).
3. **Navigate** to the `promptfoo/` directory:

   ```bash
   cd promptfoo
   ```
4. **Run the evaluation:**

   ```bash
   npx promptfoo eval -c tests/model_tests/test_file.yaml -o output/results.json --description "short_test_description"
   ```

   Example:

   ```bash
   npx promptfoo eval -c tests/codellama_tests/codellama_code_tests.yaml -o output/results.json --description "codellama test run"
   ```

>  If the test config references models like `deepseek-v3`, `remote`, or any model that depends on a local server, ensure that the model server is **running** before executing the test.

---

### Common Flags

| Flag            | Alias | Required? | Description                                             |
| --------------- | ----- | --------- | ------------------------------------------------------- |
| `--config`      | `-c`  | ✅ Yes     | Path to the test YAML file or suite.                    |
| `--output`      | `-o`  | ✅ Yes     | Output path for storing results as JSON.                |
| `--description` | —     | ❌ No      | A short label that appears in your results file and UI. |
| `--verbose`     | —     | ❌ No      | Prints detailed logs for debugging and tracing.         |
| `eval`          | —     | ✅ Yes     | Runs the evaluation with the given config.              |

---

### Viewing Test Results (Web UI)

1. **SSH into** `teach.cs.toronto.edu` using port forwarding:

   ```bash
   ssh -L 1550:localhost:1550 kukkamal@teach.cs.toronto.edu
   ```

2. **Navigate to the repo root** on the server:

   ```bash
   cd ~/Markus-Projects/ai-autograder-feedback
   ```

3. **Start the Promptfoo viewer:**

   ```bash
   npx promptfoo view --port 1550
   ```

4. **In your browser**, open [http://localhost:1550](http://localhost:1550) to view test runs.

---

### Understanding a Promptfoo Test Config (`.yaml`)

Promptfoo uses YAML configuration files to define test scenarios for evaluating model outputs. Below is a breakdown of how to read and interpret a typical test file:

#### Example: `tests/codellama_tests/codellama_code_tests.yaml`

```yaml
providers:
  - id: 'file://../../promptfoo_test_runner.py'
```

* **`providers`** defines which model(s) or wrapper(s) will be used.
* In this case, we're using a custom local Python script (`promptfoo_test_runner.py`) to run and score the responses.

---

```yaml
defaultTest:
  vars:
    model: deepSeek-R1:70B
    scope: code
    submission_type: python
    system_prompt: code_feedback_v3
```

* **`defaultTest.vars`** defines default variables for all tests in the file.

  * `model`: The model being tested.
  * `scope`: Type of task (`code`, `text`, etc.).
  * `submission_type`: Language or file type being evaluated.
  * `system_prompt`: Optional system prompt to provide context to the model.

---

```yaml
scenarios:
  - config:
      - vars: { prompt: code_feedback_r1 }
```

* **`scenarios`** define variations of tests under a shared configuration.
* Each `config.vars.prompt` refers to the prompt template ID used for this scenario (e.g., `code_feedback_r1`).
* Optionally, you can specify a `system_prompt` to provide additional context to the model.

---

```yaml
    tests:
      - description: "CSC263 correct submission"
        vars:
          submission_file: test_submissions/csc263_opt_connected/correct_submission/correct_submission.py
          solution_file:   test_submissions/csc263_opt_connected/solution.py
```

* **`tests`** is a list of test cases under the current scenario.
* Each `vars` block defines specific input files:

  * `submission_file`: The student submission being graded.
  * `solution_file`: The instructor reference solution (used for comparison but never shown to the model).
  * `description`: Optional human-readable label for the test case.

You can define multiple tests under one scenario. The prompt will run on each `submission_file`, and the test runner can score the result based on differences from the `solution_file`.

---

### Workflow Summary

For each test:

1. The prompt (e.g., `code_feedback_r1`) is invoked with the variables substituted in.
2. The model (like `deepSeek-R1:70B`) processes the prompt.
3. The output is evaluated or saved for comparison via custom logic (via `promptfoo_test_runner.py`).

---

### Tips for Writing New Tests

* Add new tests by copying an existing test block and changing the file paths.
* Organize submissions by assignment or course in the `test_submissions/` folder.
* Group related tests into a single YAML to keep evaluations coherent.
