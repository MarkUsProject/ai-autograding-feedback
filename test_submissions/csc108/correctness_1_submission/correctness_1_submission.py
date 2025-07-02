def fizzbuzz(n: int) -> list:
    """
    Generates the FizzBuzz sequence from 1 to n.

    For each number in the range 1 to n (inclusive):
    - If divisible by 3, returns 'Fizz'
    - If divisible by 5, returns 'Buzz'
    - If divisible by both 3 and 5, returns 'FizzBuzz'
    - Otherwise, returns the number itself
    """
    result = []
    for i in range(1, n + 1):
        if i % 3 == 0:
            result.append('Fizz')
        elif i % 5 == 0:
            result.append('Buzz')
        else:
            result.append(i)
    return result
