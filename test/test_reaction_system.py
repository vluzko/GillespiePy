from typing import NamedTuple, List

import numpy as np
import pytest

import gillespiepy.pysces_parser as PyscesParser
from gillespiepy.constants import MODEL_DIRECTORY

ReactionData = NamedTuple("ReactionData", (
    ("number_of_reactions", int),
    ("reaction_names", List[str]),
    ("full_matrix", np.ndarray),
    ("reactant_indices", List[List[int]]),
    ("product_indices", List[List[int]])
))


SpeciesData = NamedTuple("SpeciesData", (
    ("number_of_species", int),
    ("species_names", List[str]),
    ("species_initial", List[float])
))


SystemData = NamedTuple("SystemData", (
    ("reaction_data", ReactionData),
    ("species_data", SpeciesData)
))


@pytest.fixture
def systems():
    signaling_species = SpeciesData(15, signaling_species_names, signaling_species_initial)
    signaling_reactions = ReactionData(32,
                                       signaling_reaction_names,
                                       signaling_full_matrix,
                                       signaling_reactant_indices,
                                       signaling_product_indices)
    signaling_data = SystemData(signaling_reactions, signaling_species)

    return {
        "Signaling3cCD.psc": signaling_data
    }


def test_signaling(systems):
    system_data = systems["Signaling3cCD.psc"]
    system = PyscesParser.parse("Signaling3cCD.psc", MODEL_DIRECTORY)

    # Test reactions
    assert system.number_of_reactions == system_data.reaction_data.number_of_reactions
    assert list(system.reactions.keys()) == system_data.reaction_data.reaction_names
    assert np.array_equal(system.output_matrix - system.input_matrix, system_data.reaction_data.full_matrix)
    # assert system.reactant_indices == system_data.reaction_data.reactant_indices
    # assert system.product_indices == system_data.reaction_data.product_indices

    # Test species
    assert system.number_of_species == system_data.species_data.number_of_species
    assert list(system.species_frame.index) == system_data.species_data.species_names
    assert np.array_equal(system.species, system_data.species_data.species_initial)


signaling_full_matrix = np.array([
    [-1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.],
    [0., -1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.],
    [0.,  0., -1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.],
    [0.,  0., -1.,  0., -1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.],
    [0.,  0.,  0., -1., -1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.],
    [0.,  1.,  0.,  0.,  0., -1.,  0.,  1.,  0.,  0.,  0.,  0.,  0., 0.,  0.],
    [1.,  0.,  0.,  0.,  0.,  0., -1.,  1.,  0.,  0.,  0.,  0.,  0., 0.,  0.],
    [-1.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  1.,  0.,  0.,  0.,  0., 0.,  0.],
    [1.,  0.,  0.,  0.,  1.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0., 0.,  0.],
    [0.,  0.,  0.,  0.,  0., -1.,  1.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.],
    [0.,  0.,  0.,  0.,  1.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0., 0.,  0.],
    [1.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.],
    [0.,  1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.],
    [1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.],
    [0.,  0.,  1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.],
    [0.,  0.,  1.,  0.,  1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.],
    [0.,  0.,  0.,  1.,  1.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.],
    [0., -1.,  0.,  0.,  0.,  1.,  0., -1.,  0.,  0.,  0.,  0.,  0., 0.,  0.],
    [-1.,  0.,  0.,  0.,  0.,  0.,  1., -1.,  0.,  0.,  0.,  0.,  0., 0.,  0.],
    [1.,  0.,  0.,  0.,  0.,  0.,  0.,  1., -1.,  0.,  0.,  0.,  0., 0.,  0.],
    [0.,  0.,  0.,  0.,  0.,  1., -1.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,  0., 0.,  0.],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  1.,  0., 0.,  0.],
    [0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0., 0.,  0.],
    [1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0., 0.,  0.],
    [0.,  0.,  0.,  0.,  0.,  0.,  0., -2.,  0.,  0.,  0.,  0.,  1., 0.,  0.],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  2.,  0.,  0.,  0.,  0., -1., 0.,  0.],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1., -1., 1.],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1., 1., -1.],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,  0., 0.,  0.]
])
signaling_species_names = ['S', 'SL', 'SLP', 'SP', 'R', 'SLPR', 'SPR', 'RP', 'RPS', 'mRNA_R', 'mRNA_Sprelim', 'mRNA_S',
                           'RPDimer', 'DNA', 'DNAa']
signaling_species_initial = np.array([0., 0., 100., 0., 0., 0., 0., 80., 0., 5., 0., 5., 0., 1., 0.])
signaling_reaction_names = ['R1f', 'R2', 'R3f', 'R4f', 'R5f', 'R6f', 'R7f', 'R8f', 'R9', 'R10f', 'R11',
                            'R12', 'R13', 'R1b', 'R3b', 'R4b', 'R5b', 'R6b', 'R7b', 'R8b', 'R10b',
                            'Rp1a', 'Rp1b', 'Rp3a', 'Rd1a', 'Rp3b', 'Rd1b', 'RDimerf', 'RDimerb',
                            'RActf', 'RActb', 'Rp2']
signaling_reactant_indices = [[0], [1], [2], [2, 4], [3, 4], [5], [6], [0, 7], [8], [5], [7], [3], [2], [1], [3], [5],
                              [6], [1, 7], [0, 7], [8], [6], [], [10], [], [9], [], [11], [7, 7], [12], [12, 13], [14],
                              []]
signaling_product_indices = [[1], [2], [3], [5], [6], [1, 7], [0, 7], [8], [0, 4], [6], [4], [0], [1], [0], [2], [2, 4],
                             [3, 4], [5], [6], [0, 7], [5], [9, 10], [11], [4], [], [0], [], [12], [7, 7], [14],
                             [12, 13], [9, 10]]
