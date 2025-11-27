from enum import Enum


def get_enum_values(enum_class: type[Enum]) -> list[str]:
    """
    Retrieve all values from an Enum class.

    Args:
        enum_class (type[Enum]): The Enum class to extract values from.

    Returns:
        list[str]: A list containing all the enum values.
    """
    return [member.value for member in enum_class]


class Prompt(Enum):
    """
    Enum representing different types of prompts for AI models.

    Each prompt corresponds to a specific kind of scope (code, image, or text).
    """

    IMAGE_COMPARE = "image_compare"
    IMAGE_ANALYZE = "image_analyze"
    IMAGE_ANALYZE_ANNOTATION = "image_analyze_annotations"
    IMAGE_STYLE = "image_style"
    IMAGE_STYLE_ANNOTATIONS = "image_style_annotations"
    IMAGE_OVERALL = "image_overall"
    IMAGE_COMPARE_OVERALL = "image_compare_overall"
    IMAGE_OVERALL_WITH_INSTRUCTION = "image_overall_with_instruction"
    IMAGE_COMPARE_WITH_INSTRUCTION = "image_compare_with_instruction"
    CODE_LINES = "code_lines"
    CODE_TEMPLATE = "code_template"
    CODE_TABLE = "code_table"
    CODE_HINT = "code_hint"
    CODE_EXPLANATION = "code_explanation"
    CODE_ANNOTATION = "code_annotation"
    CODE_FEEDBACK_V3 = "code_feedback_v3"
    CODE_FEEDBACK_R1 = "code_feedback_r1"
    CODE_FUNCTIONALITY = "code_functionality"
    CODE_OVERALL = "code_overall"
    CODE_TEST_FEEDBACK = "code_test_feedback"
    CODE_OVERALL_WITH_INSTRUCTION = "code_overall_with_instruction"
    TEXT_PDF_ANALYZE = "text_pdf_analyze"
    TEXT_ANALYZE_V3 = "text_analyze_v3"
    TEXT_ANALYZE_R1 = "text_analyze_r1"
    TEXT_EFT_PROOF = "text_eft_proof"

    def __str__(self):
        return self.value


class Scope(Enum):
    """
    Enum representing the high-level content scope for a given operation.

    This helps define whether the input is image-based, code-based, or text-based.
    """

    IMAGE = "image"
    CODE = "code"
    TEXT = "text"

    def __str__(self):
        return self.value


class Models(Enum):
    """
    Enum representing the available AI model types.
    """

    OPENAI = "openai"
    OPENAIVECTOR = "openai-vector"
    LLAMA = "llama3.2-vision:90b"
    LLAVA = "llava:34b"
    DEEPSEEK = "deepSeek-R1:70B"
    CODELLAMA = "codellama:latest"
    CLAUDE = "claude-3.7-sonnet"
    REMOTE = "remote"
    DEEPSEEKV3 = "deepSeek-v3"

    def __str__(self):
        return self.value


class FileType(Enum):
    """
    Enum representing different input file types that the application can process.
    """

    JUPYTER = "jupyter"
    PYTHON = "python"
    PDF = "pdf"

    def __str__(self):
        return self.value


class OutputTemplate(Enum):
    """
    Enum representing different output template types that the application can process.
    """

    RESPONSE_ONLY = 'response_only'
    VERBOSE = 'verbose'
    RESPONSE_AND_PROMPT = 'response_and_prompt'

    def __str__(self):
        return self.value


class SystemPrompt(Enum):
    """
    Enum representing different types of system prompts for AI models.
    """

    TEST_FEEDBACK = "student_test_feedback"
    IMAGE_STYLE_GRADER = "image_style_grader"
    IMAGE_COMPARISON_GRADER = "image_comparison_grader"
    CODE_FEEDBACK_V3 = "code_feedback_v3"
    CODE_TEST_FEEDBACK = "code_test_feedback"
    TEXT_FEEDBACK_V3 = "text_feedback_v3"

    def __str__(self):
        return self.value


class LlamaMode(Enum):
    """
    Enum representing the output mode for invoking llama.cpp.
    """

    SERVER = "server"
    CLI = "cli"

    def __str__(self):
        return self.value
