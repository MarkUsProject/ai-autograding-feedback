def fizzbuzz(n: int) -> list:
    """
    Generates the FizzBuzz sequence from 1 to n.

    For each number in the range 1 to n (inclusive):
    - If divisible by 3, returns 'Fizz'
    - If divisible by 5, returns 'Buzz'
    - If divisible by both 3 and 5, returns 'FizzBuzz'
    - Otherwise, returns the number itself

    Args:
        n (int): The upper limit of the range (inclusive).

    Returns:
        list: A list containing the FizzBuzz results from 1 to n.

    Examples:
        >>> fizzbuzz(5)
        [1, 2, 'Fizz', 4, 'Buzz']

        >>> fizzbuzz(15)
        [1, 2, 'Fizz', 4, 'Buzz', 'Fizz', 7, 8, 'Fizz', 'Buzz',
        11, 'Fizz', 13, 14, 'FizzBuzz']
    """
    result = []
    for current_number in range(1, n + 1):
        if current_number % 15 == 0:
            result.append('FizzBuzz')
        elif current_number % 3 == 0:
            result.append('Fizz')
        elif current_number % 5 == 0:
            result.append('Buzz')
        else:
            result.append(current_number)
    return result
