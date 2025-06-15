import pytest

# Assuming the student's BFS function is named bfs_ubmission.py
try:
    import bfs_submission as submission
except ImportError:
    pass

graph = {
    "A": ["B", "C"],
    "B": ["A", "D", "E"],
    "C": ["A", "F"],
    "D": ["B"],
    "E": ["B", "F"],
    "F": ["C", "E"],
}

# Correct expected output for BFS traversal
correct_output = ["A", "B", "C", "D", "E", "F"]


def capture_bfs_output(bfs_function, graph, start_node):
    """Helper function to capture BFS output"""
    import sys
    from io import StringIO

    captured_output = StringIO()
    sys.stdout = captured_output

    bfs_function(graph, start_node)

    sys.stdout = sys.__stdout__  # Restore original stdout

    return captured_output.getvalue().strip().split()


def test_student_bfs():
    """Test student's BFS function for correct behavior"""
    student_output = capture_bfs_output(submission.bfs, graph, "A")

    assert student_output == correct_output, f"Expected {correct_output}, but got {student_output}"


if __name__ == "__main__":
    pytest.main()
