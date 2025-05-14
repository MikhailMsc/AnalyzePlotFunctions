import base64


def test_img_to_str():
    with open('/Users/mihail/Documents/repos/analyzer/tests/base64/2025-05-14_19-10-50.png', 'rb') as img_file:
        encoded_str = base64.b64encode(img_file.read()).decode('utf-8')
        encoded_str = str(encoded_str).replace("b'", '').replace("'", '')
        encoded_str = f'![](data:image/png;base64, {encoded_str})'
        print(encoded_str)
