
from dotenv import load_dotenv
import json
import os
import pytest
import re
import subprocess
import sys


ANNOTATION_PROMPT = """These are the student mistakes you previously identified in the 
last message. For each of the mistakes you identified, return a JSON object containing 
an array of annotations, referencing the student's submission file for line and column #s. 
Each annotation should include: filename: The name of the student's file. content: 
A short description of the mistake. line_start and line_end: The line number(s) where the
mistake occurs. Ensure the JSON is valid and properly formatted. Here is a sample format 
of the json array to return: { \"annotations\": [{\"filename\": \"student_code.py\", 
\"content\": \"Variable 'x' is unused.\", \"line_start\": 5, \"line_end\": 5,]}. 
ONLY return the json object and nothing else. Make sure the line #s don't exceed 
the number of lines in the file. You can use markdown syntax in the annotation's content,
especially when denoting code."""
    
def add_annotation_columns(annotations, submission):
    """
    Given LLM annotations with line_start and line_end, add 
    column_start and column_end metadata to the JSON object. 
    """
    try:
        file_path = submission.__file__
        with open(file_path, 'r') as file:
            file_lines = file.readlines()
    except Exception as e:
        print(f"Error reading submission file: {e}")
    
    annotations_with_columns = []

    for annotation in annotations:
        filename = annotation["filename"]
        line_start = annotation["line_start"]
        line_end = annotation["line_end"]

        if not file_lines or line_start > len(file_lines) or line_end > len(file_lines):
            print(f"Skipping invalid line numbers for {filename}: {line_start}-{line_end}")
            continue

        column_starts = []
        column_ends = []

        for i in range(line_start - 1, line_end):
            if i >= len(file_lines):
                continue

            line = file_lines[i]
            stripped_line = line.rstrip("\n")

            if stripped_line.strip(): # find start and end of each line
                start_col = len(line) - len(line.lstrip())
                end_col = len(stripped_line)
            else:
                start_col = 0
                end_col = 1

            column_starts.append(start_col)
            column_ends.append(end_col)

        if column_starts and column_ends:
            # find min and max columns if multiline annotation
            column_start = min(column_starts)
            column_end = max(column_ends)
        else:
            column_start = 0
            column_end = 1


        annotation["column_start"] = column_start
        annotation["column_end"] = column_end
        annotations_with_columns.append(annotation)

    return annotations_with_columns



def run_llm(submission, model, scope, output, question=None, prompt_text=None,prompt=None) -> str:
    """Run the LLM feedback generator and return results"""
    load_dotenv()
    llm_command = [
        sys.executable, 
        "-m", "ai_feedback",
        "--submission_type", submission,
        "--scope", scope,
        "--assignment", "./",
        "--model", model,
        "--output", output
    ]
    if question is not None:
        llm_command.append("--question")
        llm_command.append(question)
    if prompt is not None:
        llm_command.append("--prompt")
        llm_command.append(prompt)
    if prompt_text is not None:
        llm_command.append("--prompt_text")
        llm_command.append(prompt_text)
        
    # Capture the output from the LLM program
    llm_result = subprocess.run(llm_command, capture_output=True, text=True)
    try:
        llm_result.check_returncode()
    except:
        error = llm_result.stderr.strip()
        return str(f"Error calling LLM API:\n{error}")

    llm_output = llm_result.stdout.strip()
    return llm_output

def extract_json(response: str) -> list[dict]:
    """Returns a list of JSON objects found in a string"""
    matches = re.findall(r'(\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\})', response)
    return [json.loads(match) for match in matches]

MINIMUM_ANNOTATION_WIDTH = 8
def convert_coordinates(box: list[int]) -> tuple[int]:
    """Extend a bounding box to have a minimum width for image annotations"""
    x_extension = max(0, (MINIMUM_ANNOTATION_WIDTH - abs(box[2] - box[0])) // 2)
    y_extension = max(0, (MINIMUM_ANNOTATION_WIDTH - abs(box[3] - box[1])) // 2)
    return (
        box[0] - x_extension,
        box[1] - y_extension,
        box[2] + x_extension,
        box[3] + y_extension
    )

def add_image_annotations(request, llm_feedback: str, file_name: str) -> None:
    """Add image annotations from LLM feedback to the Pytest request"""
    annotations = extract_json(llm_feedback)
    for annotation in annotations:
        if "location" in annotation and "description" in annotation:
            x1, y1, x2, y2 = convert_coordinates(annotation["location"])
            request.node.add_marker(pytest.mark.markus_annotation(
                type="ImageAnnotation",
                filename=os.path.relpath(file_name, os.getcwd()),
                content=annotation["description"],
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            ))