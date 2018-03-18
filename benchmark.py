import os
import ipdb
import time
import numpy as np
from random import shuffle
import stochpy

import gillespiepy.pysces_parser as parser
from gillespiepy.constants import MODEL_DIRECTORY

timesteps = 100000


class Runner(object):

    def __init__(self):
        self.times = []

    def time_run(self, model_file):
        simulator = self.create_simulator(model_file)
        start = time.time()
        self.run(simulator)
        end = time.time()
        elapsed = end - start
        print(elapsed)
        self.times.append(elapsed)

    def create_simulator(self, model_file):
        raise NotImplementedError

    def run(self, simulator):
        raise NotImplementedError


class StochpyRunner(Runner):

    def create_simulator(self, model_file):
        simulator = stochpy.SSA()
        simulator.Model(model_file)
        return simulator

    def run(self, simulator):
        simulator.DoStochSim(end=timesteps, mode='steps')


class GillespieRunner(Runner):

    def __init__(self, directory):
        self.directory = directory
        super(GillespieRunner, self).__init__()

    def create_simulator(self, model_file):
        return parser.parse(model_file, self.directory)

    def run(self, simulator):
        species, rates = simulator.run_simulation(timesteps)


def main():
    example_dir = "gillespiepy/examples"

    models = [x for x in os.listdir(example_dir) if not x.endswith("xml")]
    shuffle(models)
    runners = (GillespieRunner(example_dir), )
    for i, model in enumerate(models):
        if model != "TranscriptionIntermediate.psc":
            continue
        print("\nCurrent model: {}\n".format(model))
        if model in ("Polymerase.psc", "GeneDuplication.psc", "chain50.psc"):
            continue
        for runner in runners:
            runner.time_run(model)


if __name__ == "__main__":
    main()
