from os.path import join, expanduser

LOCAL_DIRECTORY = join(expanduser('~'), '.gillespiepy')
MODEL_DIRECTORY = join(LOCAL_DIRECTORY, 'pscmodels')
TEMP_DIRECTORY = join(LOCAL_DIRECTORY, 'temp')

MATHML_TO_NUMPY_SYMBOLS = {
    'notanumber': 'numpy.NaN',
    'pi': 'numpy.pi',
    'infinity': 'numpy.Infinity',
    'exponentiale': 'numpy.e',
    'true': 'True',
    'false': 'False',
    'True': 'True',
    'False': 'False'
}

MATHML_TO_NUMPY_FUNCTIONS = {
    'pow': 'pow', 'root': 'pow', 'abs': 'abs',
    'exp': 'math.exp', 'ln': 'math.log', 'log': 'math.log10',
    'floor': 'numpy.floor', 'ceiling': 'numpy.ceil', 'factorial': None,
    'sin': 'numpy.sin', 'cos': 'numpy.cos', 'tan': 'numpy.tan',
    'sec': None, 'csc': None, 'cot': None,
    'sinh': 'numpy.sinh', 'cosh': 'numpy.cosh', 'tanh': 'numpy.tanh',
    'sech': None, 'csch': None, 'coth': None,
    'arcsin': 'numpy.arcsin', 'arccos': 'numpy.arccos', 'arctan': 'numpy.arctan',
    'arcsec': None, 'arccsc': None, 'arccot': None,
    'arcsinh': 'numpy.arcsinh', 'arccosh': 'numpy.arccosh', 'arctanh': 'numpy.arctanh',
    'arcsech': None, 'arccsch': None, 'arccoth': None,
    'eq': 'operator.eq', 'neq': 'operator.ne',
    'gt': 'operator.gt', 'geq': 'operator.ge',
    'lt': 'operator.lt', 'leq': 'operator.le',
    'ceil': 'numpy.ceil', 'sqrt': 'numpy.sqrt',
    'equal': 'operator.eq', 'not_equal': 'operator.ne',
    'greater': 'operator.gt', 'greater_equal': 'operator.ge',
    'less': 'operator.lt', 'less_equal': 'operator.le',
    'ne': 'operator.ne', 'ge': 'operator.ge', 'le': 'operator.le',
    'xor': 'operator.xor', 'piecewise': 'self._piecewise_', '_piecewise_': 'self._piecewise_',
    'not': 'operator.not_', 'not_': 'operator.not_'
}

UNITS_DICTIONARY = {
    'substance': {
        'exponent': 1, 'multiplier': 1.0, 'scale': 0, 'kind': 'mole'
    }, 'volume': {
        'exponent': 1, 'multiplier': 1.0, 'scale': 0, 'kind': 'litre'
    }, 'time': {
        'exponent': 1, 'multiplier': 1.0, 'scale': 0, 'kind': 'second'
    }, 'length': {
        'exponent': 1, 'multiplier': 1.0, 'scale': 0, 'kind': 'metre'
    }, 'area': {
        'exponent': 2, 'multiplier': 1.0, 'scale': 0, 'kind': 'metre'
    }
}

OPERATOR_MAPPING = {
    "+": "numpy.add",
    "-": "numpy.subtract",
    "*": "numpy.multiply",
    "/": "numpy.divide"
}
