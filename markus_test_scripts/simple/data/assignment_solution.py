def add_all(numbers: list[int]) -> int:
    """Return the sum of all of the given numbers."""
    result = 0
    for number in numbers:
        result += number

    return result
