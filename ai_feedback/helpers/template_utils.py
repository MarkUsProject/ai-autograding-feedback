import os
from pathlib import Path

import PyPDF2
from ollama import Image
from PIL import Image as PILImage


def render_prompt_template(
    prompt_content: str,
    assignment_files: list[str] = None,
    has_submission_image: bool = False,
    has_solution_image: bool = False,
    **kwargs,
) -> str:
    """Render a prompt template by replacing placeholders with actual values.

    Args:
        prompt_content (str): The prompt template with placeholders like {file_contents}
        assignment_files (list[str], optional): List of file paths for file-based placeholders
        has_submission_image (bool): Whether a submission image is present
        has_solution_image (bool): Whether a solution image is present
        **kwargs: Additional key-value pairs for placeholder replacement

    Returns:
        str: The rendered prompt with placeholders replaced
    """
    template_data = kwargs.copy()

    if assignment_files is not None:
        if '{file_references}' in prompt_content and 'file_references' not in template_data:
            template_data['file_references'] = gather_file_references(assignment_files)

        if '{file_contents}' in prompt_content and 'file_contents' not in template_data:
            template_data['file_contents'] = gather_file_contents(assignment_files)

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

    return prompt_content.format(**template_data)


def gather_file_references(assignment_files: list[str]) -> str:
    """Generate file reference descriptions for prompt templates.

    Args:
        assignment_files (list[str]): List of file paths to process

    Returns:
        str: File reference descriptions like "The instructor's solution file..."
    """
    references = []

    for file_path in assignment_files:
        filename = os.path.basename(file_path)
        name_without_ext, _ = os.path.splitext(filename)

        if name_without_ext.endswith("_solution"):
            references.append(f"The instructor's solution file you should reference is {filename}.")
        elif name_without_ext.endswith("_submission"):
            references.append(f"The student's code submission file you should reference is {filename}.")
        elif name_without_ext.endswith("test_output"):
            references.append(f"The student's error trace file you should reference is {filename}.")

    return "\n".join(references)


def gather_file_contents(assignment_files: list[str]) -> str:
    """Generate file contents with line numbers for prompt templates.

    Args:
        assignment_files (list[str]): List of file paths to process

    Returns:
        str: File contents formatted with line numbers
    """
    file_contents = ""

    for file_path in assignment_files:
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
        from ..helpers.image_reader import read_solution_images, read_submission_images

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
