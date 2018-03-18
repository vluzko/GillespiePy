from collections import namedtuple
from typing import NamedTuple, Dict, List, Tuple


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
