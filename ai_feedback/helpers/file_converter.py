import os
from pathlib import Path

import nbformat


def rename_files(file_path: str) -> str:
    """
    Renames a .ipynb file to .txt and converts it to a text file.

    Args:
        file_path (str): Path to the .ipynb file.

    Returns:
        str: Path to the converted .txt file.
    """
    base, ext = os.path.splitext(file_path)
    if ext.lower() == ".ipynb":
        renamed_file_path = base + ".txt"
        return convert_ipynb_to_txt(file_path, renamed_file_path)
    return file_path


def convert_ipynb_to_txt(ipynb_file_path: str, output_txt_file_path: str) -> str:
    """
    Converts a Jupyter notebook (.ipynb) file to a text (.txt) file.

    Args:
        ipynb_file_path (str): Path to the .ipynb file.
        output_txt_file_path (str): Path where the output .txt file will be saved.

    Returns:
        str: Path to the converted .txt file.
    """
    # Load the .ipynb file
    with open(ipynb_file_path, "r") as f:
        notebook = nbformat.read(f, as_version=4)

    # Open the output text file
    with open(output_txt_file_path, "w") as output_file:
        # Extract code and markdown cells
        for cell in notebook.cells:
            if cell.cell_type == "code":
                output_file.write(f"Code Cell:\n{cell.source}\n\n")
            elif cell.cell_type == "markdown":
                output_file.write(f"Markdown Cell:\n{cell.source}\n\n")

    return output_txt_file_path
