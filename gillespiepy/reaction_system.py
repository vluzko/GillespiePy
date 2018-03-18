import pandas as pd
import numpy as np
from typing import Dict, Tuple, Callable, Optional
from numba import jit

from gillespiepy.algorithms import algorithms_mapping


import ipdb


class ExpressionError(Exception):
    pass


class Reaction:

    def __init__(self,
                 name: str,
                 inputs: Dict[str, float],
                 outputs: Dict[str, float],
                 compartment,
                 rate_equation: str):
        """
        Creates a reaction. Mostly used by PyscesParser.parse
        :param name:            - The name of the reaction.
        :param inputs:          - A map from names of inputs to their coefficients
        :param outputs:         - A map from names of outputs to their coefficients
        :param compartment:     - The compartment the reaction takes place in (unimplemented)
        :param rate_equation:   - A string which calculates the reaction rate for this reaction when passed to eval.
        """
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.compartment = compartment
        self.rate_equation = rate_equation


class ReactionSystem:
    """

    Attributes:
        parameters:
        reactions:
        species_frame:
        constants_frame:
        species:
        constants:
        input_matrix:
        output_matrix:
        number_of_reactions:
        number_of_species:
        use_jit:
        rate_function:
    """

    def __init__(self, parameters: pd.DataFrame, reactions: Dict[str, Reaction], use_jit: bool=True,
                 rate_function: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]]=None):
        """

        Args:
            parameters:
            reactions: The reactions in the system.
            use_jit: Whether or not to optimize the rate function with numba.
            rate_function: An optional custom rate function.
        """
        self.parameters = parameters
        self.reactions = reactions
        self.species_frame, self.constants_frame = self.split_parameters()
        self.species = np.array(self.species_frame['initial'])
        self.constants = np.array(self.constants_frame['initial'])
        self.input_matrix, self.output_matrix = self.create_stoichiometry_matrices()

        for reaction in self.reactions.values():
            reaction.rate_equation = self.rewrite_expressions(reaction.rate_equation)
        self.rate_expressions = ["rates[{}] = ".format(i) + r.rate_equation for i, r in enumerate(self.reactions.values())]
        self.number_of_reactions = len(self.reactions)
        self.number_of_species = len(self.species_frame)
        self.use_jit = use_jit
        self._passed_function = rate_function
        self.rate_function = self.get_rate_function()

    def split_parameters(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Splits the parameters data frame into species and constants.

        Returns:
            The species and constants data frames.
        """
        reagent_filter = self.parameters['is_reagent']
        species_frame = self.parameters[reagent_filter][['name', 'initial', 'initialized', 'compartment']]
        constants_frame = self.parameters[~reagent_filter][['name', 'initial', 'initialized', 'compartment']]
        return species_frame, constants_frame

    def create_stoichiometry_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create the matrices of inputs and outputs of reactions.

        Returns:
            The input and output matrices
        """
        input_matrix = np.zeros((len(self.reactions), len(self.species_frame)))
        output_matrix = np.copy(input_matrix)

        for reaction_index, reaction in enumerate(self.reactions.values()):
            # Fill out input matrix
            for species, coefficient in reaction.inputs.items():
                species_index = self.species_frame.index.get_loc(species)
                input_matrix[reaction_index, species_index] = coefficient

            # Fill out output matrix
            for species, coefficient in reaction.outputs.items():
                species_index = self.species_frame.index.get_loc(species)
                output_matrix[reaction_index, species_index] = coefficient

        return input_matrix, output_matrix

    def name_to_constant(self, name: str):
        if name in self.constants_frame.index:
            new_name = str(self.constants_frame.loc[name]['initial'])
        elif name in self.species_frame.index:
            new_name = "species[{}]".format(self.species_frame.index.get_loc(name))
        else:
            new_name = name
        return new_name

    # TODO: Rewrite to use depth-first search, since Python doesn't like recursion.
    # TODO: Ideally rewrite this to use actual types.
    def rewrite_expressions(self, expression) -> str:
        """Rewrites all expression by replacing constants with their values, and species with array accesses.

        Args:
            expression: An AST representation of an expression.

        Returns:
            The expression rewritten as a single string, with all variable names changed to avoid namespace
            collisions.

        """

        if type(expression) == str:
            # A value or name.
            return self.name_to_constant(expression)
        elif len(expression) == 2:
            # A function call. Rewrite the arguments list, then return.
            func_name = expression[0]
            replaced_arguments = ','.join([self.rewrite_expressions(x) for x in expression[1]])
            return "{}({})".format(func_name, replaced_arguments)
        elif len(expression) == 3:
            if type(expression[1]) == list:
                # An expression wrapped in parentheses
                try:
                    assert expression[0] == "("
                    assert expression[2] == ")"
                except AssertionError:
                    raise ExpressionError("Expected parentheses wrapped expression. Got: {}".format(expression))
                return "({})".format(self.rewrite_expressions(expression[1]))
            else:
                # A binary operation
                left_side = self.rewrite_expressions(expression[0])
                operator = expression[1]
                right_side = self.rewrite_expressions(expression[2])
                return "{}{}{}".format(left_side, operator, right_side)
        else:
            raise ExpressionError("Error in expression: {}. Expressions must be")

    def full_expressions(self):
        full = []
        for index, rate_expression in enumerate(self.rate_expressions):
            full.append("rates[{}] = {}".format(index, rate_expression))
        return '\n'.join(full)

    def rate_function_from_expressions(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """Create a rate function wrapped with

        Returns:

        """

        strings = [
            "def calculate_rates(rates, species):\n"
        ]
        for index, expression in enumerate(self.rate_expressions):
            strings.append("  " + expression + "\n")
        strings.append("  return rates\n")
        exec(''.join(strings))
        return locals()['calculate_rates']

        # return func.calculate_rates

    def get_rate_function(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:

        if self._passed_function:
            base_function = self._passed_function
        else:
            base_function = self.rate_function_from_expressions()

        if self.use_jit:
            return jit(nopython=True)(base_function)
        else:
            return base_function

    def run_simulation(self, end: int, method: str="direct", mode='steps'):
        simulation_algorithm = algorithms_mapping[method]
        if mode == 'steps':
            # TODO: Set to max float
            end_time = 10**10
            end_steps = end
        else:
            end_time = end
            # TODO: Set to max int
            end_steps = 10**10

        return simulation_algorithm(end_time, end_steps, self.rate_function, np.empty(len(self.rate_expressions)), self.species.copy(), self.output_matrix - self.input_matrix)
