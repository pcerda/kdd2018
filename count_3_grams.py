"""
Fast counting of 3-grams for short strings.


Quick benchmarking seems to show that pure Python code is faster when
for strings less that 1000 characters, and numpy versions is faster for
longer strings.

Very long strings would benefit from probabilistic counting (bloom
filter, count min sketch) as implemented eg in the "bounter" module.

Run unit tests with "pytest count_3_grams.py"
"""

###############################################################################
from collections import Counter


def number_of_unique_3grams(string):
    """ Return the number of different tri-grams in a string
    """
    # Pure Python implementation: no numpy
    string = '$$' + string + '$$'
    return len(list(zip(string, string[1:], string[2:])))


def number_of_3grams(string):
    """ Return the number of different tri-grams in a string
    """
    # Pure Python implementation: no numpy
    string = '$$' + string + '$$'
    return len(string) - 3 + 1


def get_unique_3grams(string):
    """ Return the set of different tri-grams in a string
    """
    # Pure Python implementation: no numpy
    string = '$$' + string + '$$'
    return set(zip(string, string[1:], string[2:]))


def get_3grams(string):
    """ Return the set of different tri-grams in a string
    """
    # Pure Python implementation: no numpy
    string = '$$' + string + '$$'
    return list(zip(string, string[1:], string[2:]))


def dictionary_of_3grams(strings):
    """ Given a list of strings, return a dictionary with key
    all the unique grams and with values a list of index indicating
    the strings that contain the respective ngram.
    """
    ngram_dict1 = {}
    for i, string in enumerate(strings):
        ngrams = get_3grams(string)
        for ngram in ngrams:
            ngram_dict1.setdefault(ngram, {})
            ngram_dict1[ngram].setdefault(i, 0)
            ngram_dict1[ngram][i] += 1
    return ngram_dict1


def strings_length(strings):
    """ Given a list of strings, return a vector with the respective lenght
    of each string in the list
    """
    strings_len = np.array([number_of_3grams(string)
                            for string in strings])
    return strings_len


def ngram_similarity(string1, strings_len, gram_dict):
    string1_ngrams = get_3grams(string1)
    count_string1_ngrams = {s: string1_ngrams.count(s)
                            for s in set(string1_ngrams)}
    samegrams = np.zeros(len(strings_len)).astype(float)
    for gram in count_string1_ngrams:
        try:
            samegrams[list(gram_dict[gram]
                      .keys())] += np.minimum(np.array(list(gram_dict[gram]
                                              .values())),
                                              count_string1_ngrams[gram])
        except KeyError:
            continue
    allgrams = number_of_3grams(string1) + strings_len - samegrams
    similarity = samegrams/allgrams
    return similarity


def number_of_common_3grams(string1, string2):
    """ Return the number of common tri-grams in two strings
    """
    # Pure Python implementation: no numpy
    tri_grams = set(zip(string1, string1[1:], string1[2:]))
    tri_grams = tri_grams.intersection(zip(string2, string2[1:],
                                           string2[2:]))
    return len(tri_grams)


def test_number_of_3grams():
    # Small unit tests
    assert number_of_3grams('abc') == 1
    for i in range(1, 7):
        assert number_of_3grams(i * 'aaa') == 1
        assert number_of_3grams('abcdefghifk'[:i+2]) == i


def test_number_of_common_3grams():
    # Small unit tests
    for i in range(1, 7):
        assert number_of_common_3grams(i * 'aaa', i * 'aaa') == 1
        assert number_of_common_3grams('abcdefghifk'[:i+2],
                                       'abcdefghifk'[:i+2]) == i


###############################################################################
# Numpy versions
import numpy as np

def number_of_3grams_np(string):
    """ Return the number of different tri-grams in a string
    """
    arr = np.frombuffer(string, dtype='S1')

    # Hard coding the 3 gram
    # Appending 3 shifted versions of the strings
    arr = np.concatenate((arr[:-2, None], arr[1:-1, None], arr[2:, None]),
                         axis=1)
    # "concatenate" strings
    arr = arr.view('S3')
    return np.unique(arr).size


def number_of_common_3grams_np(string1, string2):
    """ Return the number of common tri-grams in two strings
    """
    arr1 = np.frombuffer(string1, dtype='S1')
    arr2 = np.frombuffer(string2, dtype='S1')

    # Hard coding the 3 gram
    # Appending 3 shifted versions of the strings
    arr1 = np.concatenate((arr1[:-2, None], arr1[1:-1, None], arr1[2:, None]),
                          axis=1)
    # "concatenate" strings
    arr1 = arr1.view('S3')

    arr2 = np.concatenate((arr2[:-2, None], arr2[1:-1, None], arr2[2:, None]),
                          axis=1)
    arr2 = arr2.view('S3')

    return np.lib.arraysetops.intersect1d(arr1, arr2).size


def test_number_of_3grams_np():
    # Small unit tests
    assert number_of_3grams_np('abc') == 1
    for i in range(1, 7):
        assert number_of_3grams_np(i * 'aaa') == 1
        assert number_of_3grams_np('abcdefghifk'[:i+2]) == i


def test_number_of_common_3grams_np():
    # Small unit tests
    for i in range(1, 7):
        assert number_of_common_3grams_np(i * 'aaa', i * 'aaa') == 1
        assert number_of_common_3grams_np('abcdefghifk'[:i+2],
                                          'abcdefghifk'[:i+2]) == i

###############################################################################

if __name__ == '__main__':
    # Our two example strings
    s1 = 'patricio'
    s2 = 'patricia'

    print(number_of_common_3grams(s1, s2))
