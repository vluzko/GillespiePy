import cProfile
import pstats

import gillespiepy.pysces_parser as parser
from gillespiepy.constants import MODEL_DIRECTORY

STATS_FILE = "stats"
SYSTEM = "Signaling3cCD.psc"
SIMPLE_SYSTEM = "BirthDeath.psc"


def run_simulation():
    reaction_system = parser.parse(SYSTEM, MODEL_DIRECTORY)
    species, rates = reaction_system.run_simulation(10000)
    # print(species)


cProfile.run('run_simulation()', STATS_FILE)

stats = pstats.Stats(STATS_FILE)
stats.strip_dirs()
stats.sort_stats('tottime')
stats.print_stats(5)
