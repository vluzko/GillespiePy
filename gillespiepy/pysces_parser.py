from typing import Dict
from collections import OrderedDict
import numpy as np
import pandas as pd

import os
import copy
from ply import lex, yacc
import pdb

from gillespiepy.constants import UNITS_DICTIONARY
from gillespiepy import reaction_system 


def flatten(l: list):
    """Flattens a list iteratively."""
    flattened = []
    while l:
        first = l.pop(0)
        if isinstance(first, list):
            for element in reversed(first):
                l.insert(0, element)
        else:
            flattened.append(first)
    return flattened


class ParserException(Exception):
    pass


class PyscesParser:

    MathmlToInfix = {
        'and': 'and',
        'or': 'or',
        'true': 'True',
        'false': 'False',
        'xor': 'xor'
    }

    precedence = (
        ('left',  'PLUS', 'MINUS'),
        ('left',  'TIMES', 'DIVIDE'),
        ('left',  'POWER'),
        ('right', 'UMINUS')
    )

    # List of token names
    tokens = ('IRREV',
              'REAL',
              'INT',
              'PLUS',
              'MINUS',
              'TIMES',
              'DIVIDE',
              'POWER',
              'DOT',
              'LPAREN',
              'RPAREN',
              'EQUALS',
              'SYMBEQUI',
              'COMMA',
              'REACTION_ID',
              'NAME',
              'POOL',
              'UNIT',
              'MULTICOMMENT')

    def __init__(self):
        self.reactions = OrderedDict()  # type: OrderedDict[str, reaction_system.Reaction]
        self.units_dict = copy.deepcopy(UNITS_DICTIONARY)
        self.parameters = pd.DataFrame(columns=['name', 'initial', 'is_reagent', 'initialized', 'compartment'])
        self.parameters['initial'] = self.parameters['initial'].astype(np.float64)
        self.parameters['is_reagent'] = self.parameters['is_reagent'].astype(bool)
        self.parameters['initialized'] = self.parameters['initialized'].astype(bool)

        # Elementary regular expressions used as building blocks
        int_regex = r'\d+'
        decimal_regex = int_regex + '\.' + int_regex
        exp_regex = r'([E|e][\+|\-]?)' + int_regex
        real_regex = decimal_regex + '(' + exp_regex + ')?' + '|' + int_regex + exp_regex

        # Simple tokens
        self.t_IRREV = r'>'
        self.t_REAL = real_regex
        self.t_INT = int_regex
        self.t_PLUS = r'\+'
        self.t_MINUS = r'-'
        self.t_DIVIDE = r'/'
        self.t_TIMES = r'\*'
        self.t_POWER = '\*\*'
        self.t_DOT = '\.'
        self.t_LPAREN = r'\('
        self.t_RPAREN = r'\)'
        self.t_EQUALS = r'='
        self.t_COMMA = r','
        self.t_POOL = r'\$pool'

    t_ignore = ' \t\r'

    @staticmethod
    def t_comment(t):
        """\#.*\n"""

    @staticmethod
    def t_multiline_comment(t):
        """\"\"\"(.|\n)*?\"\"\""""
        t.type = 'MULTICOMMENT'

    @staticmethod
    def t_newline(t):
        r'\n+'
        t.lineno += len(t.value)

    @staticmethod
    def t_comparison_symbol(t):
        """!=|<"""
        t.type = 'SYMBEQUI'
        return t

    @staticmethod
    def t_unit(t):
        """Unit(Substance|Time|Length|Area|Volume):.+\n"""
        t.type = 'UNIT'
        return t

    @staticmethod
    def t_reaction_id(t):
        """[a-zA-Z_]\w*:"""
        t.type = 'REACTION_ID'
        t.value = t.value[:-1]
        return t

    @staticmethod
    def t_stoichiometric_coefficient(t):
        """STOICH_COEF: \{\d+|\d+\.\d*\} """
        t.type = "STOICH_COEF"
        return t

    @staticmethod
    def t_name(t):
        """[a-zA-Z_]\w*"""
        t.type = 'NAME'
        return t

    @staticmethod
    def t_error(t):
        print("Illegal character '{0}'".format(t.value[0]))
        t.lexer.skip(1)

    @staticmethod
    def p_error(t):
        return t

    @staticmethod
    def p_model(t):
        """Model : Statement
                 | Model Statement """

    @staticmethod
    def p_statement(t):
        """Statement : ReactionLine
                     | Assignment
                     | Unit
                     | MultiComment
                     | SymbEqui"""

    @staticmethod
    def p_inequalities_symb(t):
        """SymbEqui : SYMBEQUI"""
        t[0] = t[1]

    @staticmethod
    def p_multiline_comment(t):
        """MultiComment : MULTICOMMENT"""

    def p_unit(self, t):
        """Unit : UNIT"""
        u = t[1].split(',')
        u[0] = u[0].split(':')
        u.append(u[0][1])
        u[0] = u[0][0].lower()
        u[0] = u[0].replace('unit', '')
        for i in range(len(u)):
            u[i] = u[i].strip()
            if i in [1, 2, 3]:
                u[i] = float(u[i])
        self.units_dict[u[0]] = {
            'multiplier': u[1],
            'scale': u[2],
            'exponent': u[3],
            'kind': u[4]
        }

    # TODO: Rename to assignments
    def p_assignment(self, t):
        """Assignment : NAME EQUALS Expression"""
        name = t[1]
        expr = t[3]
        if isinstance(expr, str):
            value = eval(expr)
        elif isinstance(expr, list):
            value = eval(''.join(flatten(expr)))
        else:
            raise ParserException("RHS of an assignment must be either a value or an expression "
                                  "composed solely of values and operators. Got: {}".format(expr))

        if name in self.parameters.index:
            self.parameters.set_value(name, 'initial', value)
            self.parameters.set_value(name, 'initialized', True)
        else:
            self.parameters.loc[name] = [name, value, False, True, None]
            # t[0] = t[1] + t[2] + t[3]

    def p_reaction_line(self, t):
        """ReactionLine : REACTION_ID ReactionEq Expression"""

        reaction_name = t[1]
        # TODO: Warn if REACTION_ID already exists
        inputs, outputs = t[2]

        # Extract reagents
        compressed_inputs = {}      # type: Dict[str: float]
        compressed_outputs = {}     # type: Dict[str: float]
        # Sum all coefficients on each side, to handle cases like X+X > Y
        for reagent, coefficient in inputs:
            compressed_inputs[reagent] = coefficient + compressed_inputs.get(reagent, 0)
        for reagent, coefficient in outputs:
            compressed_outputs[reagent] = coefficient + compressed_outputs.get(reagent, 0)

        # Remove any reagents with a zero coefficient (i.e. they don't actually participate in the reaction)
        # TODO: Warn if this occurs.
        compressed_inputs = {k: v for k, v in compressed_inputs.items() if v != 0}
        compressed_outputs = {k: v for k, v in compressed_outputs.items() if v != 0}

        try:
            rate_equation = t[3]
        except IndexError:
            rate_equation = ""

        react = reaction_system.Reaction(reaction_name, compressed_inputs, compressed_outputs, None, rate_equation)
        self.reactions[reaction_name] = react

    @staticmethod
    def p_reaction_eq(t):
        """ReactionEq : HalfReaction IRREV  HalfReaction
                      | POOL IRREV  HalfReaction
                      | HalfReaction IRREV POOL"""
        if t[1] == '$pool':
            t[0] = ([], t[3])
        elif t[3] == '$pool':
            t[0] = (t[1], [])
        else:
            t[0] = (t[1], t[3])

    @staticmethod
    def p_half_reaction(t):
        """HalfReaction : Term
                        | Term PLUS HalfReaction"""
        t[0] = [t[1]]
        try:
            t[0] += t[3]
        except IndexError:
            return

    def p_term(self, t):
        """Term : Number NAME
                | NAME"""

        try:
            t[0] = (t[2], float(t[1]))
            name = t[2]
        except IndexError:
            t[0] = (t[1], 1.0)
            name = t[1]
        if name in self.parameters.index:
            self.parameters.set_value(name, 'is_reagent', True)
        else:
            self.parameters.loc[name] = [name, -1, True, False, None]

    # TODO: Add support for parenthesized expressions
    @staticmethod
    def p_rate_eq(t):
        """Expression : Expression PLUS Expression
                      | Expression MINUS Expression
                      | Expression TIMES Expression
                      | Expression DIVIDE Expression
                      | Expression POWER Expression
                      | LPAREN Expression RPAREN
                      | Number
                      | FuncCall
                      | DottedName"""
        if len(t.slice) == 4:
            t[0] = [t[1], t[2], t[3]]
        else:
            t[0] = t[1]

    @staticmethod
    def p_dotted_name(t):
        """DottedName : NAME
                      | NAME DOT DottedName"""
        t[0] = t[1]

    @staticmethod
    def p_func_call(t):
        """FuncCall : DottedName LPAREN ArgList RPAREN"""
        t[0] = [t[1], t[3]]

    @staticmethod
    def p_uminus(t):
        """Expression : MINUS Expression %prec UMINUS"""
        # Alternative '''UMINUS : MINUS Expression'''

        t[0] = t[1] + t[2]

    @staticmethod
    def p_number(t):
        """Number : REAL
                  | INT"""
        t[0] = t[1]

    @staticmethod
    def p_arg_list(t):
        """ArgList : Expression
                   | ArgList COMMA Expression"""
        if len(t) == 2:
            t[0] = [t[1]]
        elif len(t) == 4:
            t[0] = [t[1]].extend(t[3])
        else:
            pdb.set_trace()


# TODO: Just take a path.
def parse(model_file: str, model_dir: str) -> reaction_system.ReactionSystem:
    """

    Args:
        model_file: Name of the file to read the model from.
        model_dir: Name of the directory.

    Returns:

    """
    # Read model from file
    with open(os.path.join(model_dir, model_file), 'r') as file:
        model = file.read()

    # Make sure the file ends with a newline.
    if model[-1] != "\n":
        model += "\n"

    parser_model = PyscesParser()

    lexer = lex.lex(module=parser_model)
    lexer.input(model)
    parser = yacc.yacc(module=parser_model, debug=0, write_tables=0)

    while True:
        token = lexer.token()
        if not token:
            break

    while True:
        p = parser.parse(model)
        if not p:
            break

    system = reaction_system.ReactionSystem(parser_model.parameters, parser_model.reactions)

    return system
