import numpy as np
import pytest
from cnn_solution import CNN as SolCNN
from cnn_solution import convolution_2d as sol_convolution
from cnn_solution import max_pooling_2d as sol_max_pooling
from cnn_submission import (  # Importing student version
    CNN,
    convolution_2d,
    max_pooling_2d,
)


def test_convolution():
    input_image = np.random.rand(5, 5)
    kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])

    student_output = convolution_2d(input_image, kernel)
    solution_output = sol_convolution(input_image, kernel)

    assert np.allclose(student_output, solution_output), "Convolution output differs!"


def test_max_pooling():
    input_image = np.random.rand(4, 4)
    pool_size = (2, 3)

    student_output = max_pooling_2d(input_image, pool_size)
    solution_output = sol_max_pooling(input_image, pool_size)

    assert np.allclose(student_output, solution_output), "Max pooling output differs!"


def test_cnn_forward():
    input_data = np.random.rand(1, 5, 5)  # Example input
    student_cnn = CNN()
    solution_cnn = SolCNN()

    student_output = student_cnn.forward(input_data)
    solution_output = solution_cnn.forward(input_data)

    assert np.allclose(student_output, solution_output), "CNN forward pass output differs!"
