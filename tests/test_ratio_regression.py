import unittest
import numpy as np
from ratio_regression.genetic import SymbolicRegressor
from ratio_regression.functions import _FUNCTION_MAP

class TestRatioRegression(unittest.TestCase):

    def test_functions(self):
        for fun in _FUNCTION_MAP:
            print(_FUNCTION_MAP[fun])
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()