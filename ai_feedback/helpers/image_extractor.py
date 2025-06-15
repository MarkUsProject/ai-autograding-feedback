import base64
import json
import os


def extract_images(input_notebook_path: os.PathLike, output_directory: os.PathLike, output_name: str):
    with open(input_notebook_path, "r") as file:
        notebook = json.load(file)
        os.makedirs(output_directory, exist_ok=True)
        for cell_number, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                # Choosing the saved image's name
                source = cell["source"]
                if len(source) > 0 and source[0].startswith("# Question"):
                    # Cell header for file names
                    question_name = source[0][2:-1]
                elif "markus_question_name" in cell["metadata"]:
                    # Cell metadata tag for saved image file name
                    question_name = cell["metadata"]["markus_question_name"]
                else:
                    continue

                # Find images and save them
                image_count = 0
                for output in cell["outputs"]:
                    for file_type, data in output["data"].items():
                        if "image/" in file_type:
                            ext = file_type.split("/")[-1]
                            image_filename = f"{output_name}.{ext}"
                            os.makedirs(
                                os.path.join(output_directory, question_name, str(image_count)),
                                exist_ok=True,
                            )
                            image_path = os.path.join(
                                output_directory,
                                question_name,
                                str(image_count),
                                image_filename,
                            )
                            image_count += 1

                            image_data = base64.b64decode(data)
                            with open(image_path, "wb") as img_file:
                                img_file.write(image_data)

                # Save question context (source of previous cell)
                if cell_number >= 1 and notebook["cells"][cell_number - 1]["cell_type"] == "markdown":
                    question_context_data = "".join(notebook["cells"][cell_number - 1]["source"])
                    question_context_filename = "context.txt"
                    os.makedirs(os.path.join(output_directory, question_name), exist_ok=True)
                    question_context_path = os.path.join(output_directory, question_name, question_context_filename)
                    with open(question_context_path, "w") as txt_file:
                        txt_file.write(question_context_data)
