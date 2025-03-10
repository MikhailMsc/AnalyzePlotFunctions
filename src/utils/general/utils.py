from typing import Union


def get_accuracy(number) -> int:
    number = str(number)
    if '.' not in number:
        return 0
    else:
        return len(number.split('.')[-1].rstrip('0'))


def pretty_number(number: Union[int, float]) -> str:
    number = str(number)
    if '.' not in number:
        left_part, right_part = number, ''
    else:
        left_part, right_part = number.split('.')

    left_part = [
        '_' + n if i % 3 == 2 else n
        for i, n in enumerate(left_part[::-1])
    ][::-1]
    left_part = ''.join(left_part).lstrip('_')

    if right_part:
        return left_part + '.' + right_part
    else:
        return left_part
