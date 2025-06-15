Analyze the student's submission code and identify specific parts of the code which are causing mistakes, compared to the instructor's solution code. Return a JSON object containing an array of annotations. Each annotation should include: filename: The name of the student's file. content: A short description of the mistake. line_start and line_end: The line number(s) where the mistake occurs. column_start and column_end: The approximate column range of the mistake. Ensure the JSON is valid and properly formatted. Here is a sample format of the json array to return: {{ "annotations": [{{"filename": "student_code.py", "content": "Variable 'x' is unused.", "line_start": 5, "line_end": 5, "column_start": 4, "column_end": 5}}] }}. ONLY return the json object and nothing else.

{file_references}

Files to Reference:
{file_contents}
