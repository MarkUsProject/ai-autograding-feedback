import os
import sys
import re
from pathlib import Path
from string import Template
from typing import Optional, List
from PIL import Image as PILImage
from ollama import Image
import PyPDF2


def render_prompt_template(
        prompt_content: str,
        submission: Path,
        has_submission_image: bool = False,
        has_solution_image: bool = False,
        solution: Optional[Path] = None,
        test_output: Optional[Path] = None,
        question_num: Optional[int] = None,
        **kwargs) -> str:
    """Render a prompt template by replacing placeholders with actual values.

    Args:
        prompt_content (str): The prompt template with placeholders
        submission (Path): Path to the student's submission file
        solution (Path, optional): Path to the instructor's solution file
        test_output (Path, optional): Path to the student's test output or error trace file
        question_num (int, optional): The question number to use
        prompt_content (str): The prompt template with placeholders like {file_contents}
        has_submission_image (bool): Whether a submission image is present
        has_solution_image (bool): Whether a solution image is present
        **kwargs: Additional key-value pairs for placeholder replacement

    Returns:
        str: The rendered prompt with placeholders replaced
    """
    template_data = kwargs.copy()

    template_data['file_references'] = gather_file_references(submission, solution, test_output)
    if question_num is not None:
        template_data['file_contents'] = _get_question_contents([submission, solution], question_num)
    else:
        template_data['file_contents'] = gather_file_contents([submission, solution, test_output])

    if '{file_references}' in prompt_content or '{file_contents}' in prompt_content:
        return prompt_content.format(**template_data)
    if 'file_references' in template_data:
        prompt_content += "\n" + template_data['file_references']
    if 'file_contents' in template_data:
        prompt_content += "\n" + template_data['file_contents']

    # Handle image placeholders with context-aware replacement
    if '{submission_image}' in prompt_content and 'submission_image' not in template_data:
        if has_submission_image and has_solution_image:
            template_data['submission_image'] = 'The first attached image is the student\'s submission.'
        elif has_submission_image:
            template_data['submission_image'] = 'The attached image is the student\'s submission.'
        else:
            template_data['submission_image'] = '[Submission Image Attached]'

    if '{solution_image}' in prompt_content and 'solution_image' not in template_data:
        if has_submission_image and has_solution_image:
            template_data['solution_image'] = 'The second attached image is the expected solution.'
        elif has_solution_image:
            template_data['solution_image'] = 'The attached image is the expected solution.'
        else:
            template_data['solution_image'] = '[Solution Image Attached]'

    if any(tok in prompt_content for tok in ('{file_references}', '{file_contents}',
                                             '{submission_image}', '{solution_image}')):
        return Template(prompt_content).safe_substitute(**template_data)
    else:
        return prompt_content

def gather_file_references(
        submission: Path,
        solution: Optional[Path],
        test_output: Optional[Path]) -> str:
    """Generate file reference descriptions for prompt templates.

    Args:
        submission (Path): Student's submission file path
        solution (Path, optional): Instructor's solution file path
        test_output (Path, optional): Student's test output file path

    Returns:
        str: Descriptions like "The instructor's solution file..."
    """
    references: List[str] = []
    references.append(f"The student's submission file is {submission.name}.")
    if solution:
        references.append(f"The instructor's solution file is {solution.name}.")
    if test_output:
        references.append(f"The student's test output file is {test_output.name}.")
    return "\n".join(references)


def gather_file_contents(assignment_files: List[Optional[Path]]) -> str:
    """Generate file contents with line numbers for prompt templates.

    Args:
        assignment_files (list[str]): List of file paths to process
        
    Returns:
        str: File contents formatted with line numbers
    """
    file_contents = ""
    
    for file_path in assignment_files:
        if not file_path:
            continue
        filename = os.path.basename(file_path)
        
        try:
            # Handle PDF files separately
            if filename.lower().endswith('.pdf'):
                text_content = extract_pdf_text(file_path)
                file_contents += f"=== {filename} ===\n"
                lines = text_content.split('\n')
                for i, line in enumerate(lines, start=1):
                    stripped_line = line.rstrip()
                    if stripped_line.strip():
                        file_contents += f"(Line {i}) {stripped_line}\n"
                    else:
                        file_contents += f"(Line {i}) \n"
                file_contents += "\n"
            else:
                # Handle regular text files
                with open(file_path, "r", encoding="utf-8") as file:
                    lines = file.readlines()
                
                file_contents += f"=== {filename} ===\n"
                for i, line in enumerate(lines, start=1):
                    stripped_line = line.rstrip("\n")
                    if stripped_line.strip():
                        file_contents += f"(Line {i}) {stripped_line}\n"
                    else:
                        file_contents += f"(Line {i}) {line}"
                file_contents += "\n"
                
        except Exception as e:
            print(f"Error reading file {filename}: {e}")
            continue
    
    return file_contents


def extract_pdf_text(pdf_path: str) -> str:
    """Extract text content from a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text content from the PDF
    """
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
            
            return text.strip()
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {e}")
        return f"[Error: Could not extract text from PDF {os.path.basename(pdf_path)}]"


def gather_image_context(output_directory: str, question: str) -> str:
    """Gather question context for image prompts.
    
    Args:
        output_directory (str): Directory containing extracted images
        question (str): Question identifier
        
    Returns:
        str: Question context content
    """
    try:
        from ..helpers.image_reader import read_question_context
        return read_question_context(output_directory, question)
    except Exception as e:
        print(f"Error reading question context: {e}")
        return ""


def gather_image_size(output_directory: str, question: str) -> str:
    """Gather image size information for prompts.
    
    Args:
        output_directory (str): Directory containing extracted images
        question (str): Question identifier
        
    Returns:
        str: Image size in format "width by height"
    """
    try:
        from ..helpers.image_reader import read_submission_images
        submission_image_paths = read_submission_images(output_directory, question)
        if submission_image_paths:
            image = PILImage.open(submission_image_paths[0])
            return f"{image.width} by {image.height}"
    except Exception as e:
        print(f"Error reading image size: {e}")
    return "unknown"


def gather_images(output_directory: str, question: str, include_images: list[str]) -> list[Image]:
    """Gather images for attachment to message.
    
    Args:
        output_directory (str): Directory containing extracted images
        question (str): Question identifier
        include_images (list[str]): List of image types to include ("submission", "solution")
        
    Returns:
        list[Image]: List of Image objects for message attachment
    """
    images = []
    
    try:
        from ..helpers.image_reader import read_submission_images, read_solution_images
        
        if "submission" in include_images:
            submission_paths = read_submission_images(output_directory, question)
            if submission_paths:
                images.append(Image(value=submission_paths[0]))
                
        if "solution" in include_images:
            solution_paths = read_solution_images(output_directory, question)
            if solution_paths:
                images.append(Image(value=solution_paths[0]))
                
    except Exception as e:
        print(f"Error gathering images: {e}")
    
    return images

def _get_question_contents(assignment_files: List[Optional[Path]], question_num: int
    ) -> str:
        """
        Retrieve contents of files specifically for a targeted question number.

        Assumes files follow a specific markdown-like structure with sections titled
        '## Introduction' and '## Task {question_num}'.

        Args:
            assignment_files (List[Optional[Path]]): List of Path or None objects to parse.
            question_num (int): The target task number to extract.

        Returns:
            str: Combined content of the introduction and the specified task from matching files.

        Raises:
            SystemExit: If no matching task is found in the provided files.
        """
        file_contents = ""
        task_found = False

        for file_path in assignment_files:
            if (
                not file_path
                or file_path.suffix != '.txt'
                or "error_output" in file_path.name
                or file_path.name == ".DS_Store"
            ):
                continue

            content = file_path.read_text()

            intro_match = re.search(
                r"(## Introduction\b.*?)(?=\n##|\Z)", content, re.DOTALL
            )
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
