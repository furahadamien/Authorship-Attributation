# @Author: SashaChernykh
# @Date: 2018-09-01 13:31:06
# @Last Modified time: 2018-09-01 16:26:57
"""kira_encoding module."""
import codecs
import glob

import chardet

ALL_FILES = glob.glob('*.txt')

def kira_encoding_function():
    """Check encoding and convert to utf-8, if encoding no utf-8."""
    for filename in ALL_FILES:

        # Not 100% accuracy:
        # https://stackoverflow.com/a/436299/5951529
        # Check:
        # https://chardet.readthedocs.io/en/latest/usage.html#example-using-the-detect-function
        # https://stackoverflow.com/a/37531241/5951529
        with open(filename, 'rb') as opened_file:
            bytes_file = opened_file.read()
            chardet_data = chardet.detect(bytes_file)
            fileencoding = (chardet_data['encoding'])
            print('fileencoding', fileencoding)

            if fileencoding in ['UTF-8', 'us-ascii']:
                print(filename + ' in utf-8 encoding')
            else:
                # Convert file to UTF-8:
                # https://stackoverflow.com/q/19932116/5951529
                cyrillic_file = bytes_file.decode('cp1251')
                with codecs.open(filename, 'w', 'UTF-8') as converted_file:
                    converted_file.write(cyrillic_file)
                print(filename +
                      ' in ' +
                      fileencoding +
                      ' encoding automatically converted to UTF-8')


kira_encoding_function()