import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import anthropic
import PyPDF2
from dotenv import load_dotenv

from .Model import Model

# Load environment variables from .env file
load_dotenv()


class ClaudeModel(Model):
    def __init__(self) -> None:
        """
        Initializes the ClaudeModel with the Anthropic client using an API key.
        """
        super().__init__()
        self.client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))

    def generate_response(
        self,
        prompt: str,
        submission_file: Path,
        system_instructions: str,
        solution_file: Optional[Path] = None,
        scope: Optional[str] = None,
        question_num: Optional[int] = None,
        test_output: Optional[Path] = None,
        llama_mode: Optional[str] = None,
    ) -> Optional[Tuple[str, str]]:
        """
        Generates a response from Claude using the provided prompt and assignment file context.

        Args:
            prompt (str): The user's prompt for the model.
            solution_file: path to the solution file.
            submission_file: path to the submission file.
            test_output: path to the file containing the results of tests run on user submission
            scope (Optional[str]): The content scope.
            question_num (Optional[int]): Specific task number to extract from text files.
            system_instructions (str): instructions for the model
            llama_mode (Optional[str]): Optional mode to invoke llama.cpp in.

        Returns:
            Optional[Tuple[str, str]]: The original prompt and the model's response, or None if the response is invalid.
        """
        request = ""

        assignment_files = [f for f in [submission_file, solution_file, test_output] if f]

        if question_num:
            request += f" Identify and generate a response for the mistakes **only** in question/task ${question_num}. "
            file_contents = self._get_question_contents(assignment_files, question_num)
        elif scope == "text":
            file_contents = self._get_pdf_contents(submission_file, solution_file)
        else:
            file_contents = self._get_file_contents(assignment_files)

        request += f"Prompt: {prompt}\n\nFiles to Reference:\n{file_contents}"

        response = self.client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1000,
            temperature=0.5,
            system=system_instructions,
            messages=[{"role": "user", "content": request}],
        )

        if not response or not response.content:
            print("Error: Invalid or empty response from Claude.")
            return None

        return prompt, response.content[0].text

    def _get_file_contents(self, assignment_files: List[Path]) -> str:
        """
        Retrieves contents of all files and concatenates them with line numbers.

        Args:
            assignment_files (List[Path]): A list of Path Objects.

        Returns:
            str: Combined, formatted content of the files with line numbers.
        """
        file_contents = ""

        for file_path in assignment_files:
            file_name = file_path.name

            try:
                lines = file_path.read_text().splitlines()
            except Exception as e:
                print(f"Error reading file {file_name}: {e}")
                continue

            file_contents += f"=== {file_name} ===\n"
            for i, line in enumerate(lines, start=1):
                stripped_line = line.rstrip("\n")
                if stripped_line.strip():
                    file_contents += f"(Line {i}) {stripped_line}\n"
                else:
                    file_contents += f"(Line {i}) {line}"  # Keep blank lines for readability

            file_contents += "\n"

        return file_contents

    def _get_pdf_contents(self, submission_file: Path, solution_file: Path) -> str:
        """
        Retrieves text content from PDF files specifically for student and instructor submissions.

        Args:
            submission_file (Path): Path to the submission file.
            solution_file (Path): Path to the solution file.
            test_output (Path): Path to the test output file.

        Returns:
            str: Combined content of student and instructor PDFs.
        """
        student_pdf_content = ""
        instructor_pdf_content = ""

        if submission_file:
            student_pdf_content = self._extract_text_from_pdf(submission_file)
        elif solution_file:
            instructor_pdf_content = self._extract_text_from_pdf(solution_file)

        combined_content = (
            f"student_pdf_submission.pdf:\n{student_pdf_content}\n\n"
            f"instructor_pdf_solution.pdf:\n{instructor_pdf_content}"
        )

        return combined_content

    def _extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extracts and formats text from a PDF file, page by page.

        Args:
            pdf_path (Path): Path object poiting to the PDF file

        Returns:
            str: Text content extracted from the PDF.
        """
        text = ""

        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(reader.pages):
                raw_text = page.extract_text()
                if not raw_text:
                    continue

                cleaned_text = raw_text.strip()
                text += f"Page {page_num}:\n{cleaned_text}\n\n"

        return text

    def _get_question_contents(self, assignment_files: List[Path], question_num: int) -> str:
        """
        Retrieves content related to a specific question number from assignment files.

        Args:
            assignment_files (List[str]): A list of assignment file paths.
            question_num (int): The task number to extract content for.

        Returns:
            str: Combined content of introduction and matching tasks across relevant files.

        Raises:
            SystemExit: If the specified task number is not found in any file.
        """
        file_contents = ""
        task_found = False

        for file_path in assignment_files:
            if file_path.suffix != ".txt" or "error_output" in file_path.name or file_path.name == ".DS_Store":
                continue

            content = file_path.read_text()

            intro_match = re.search(r"(## Introduction\b.*?)(?=\n##|\Z)", content, re.DOTALL)
            intro_content = intro_match.group(1).strip() if intro_match else ""

            task_pattern = rf"(## Task {question_num}\b.*?)(?=\n##|\Z)"
            task_match = re.search(task_pattern, content, re.DOTALL)

            task_content = ""
            if task_match:
                task_content = task_match.group(1).strip()
                task_found = True

            file_contents += f"\n\n---\n### {file_path}\n\n"
            file_contents += intro_content + "\n\n" if intro_content else ""
            file_contents += task_content + "\n\n"

        if not task_found:
            print(f"Task {question_num} not found in any assignment file.")
            sys.exit(1)

        return file_contents.strip()
