'''
Sugarscape Constant Growback Model
================================

Replication of the model found in Netlogo:
Li, J. and Wilensky, U. (2009). NetLogo Sugarscape 2 Constant Growback model.
http://ccl.northwestern.edu/netlogo/models/Sugarscape2ConstantGrowback.
Center for Connected Learning and Computer-Based Modeling,
Northwestern University, Evanston, IL.
'''

from mesa import Model
from mesa.new_space import Grid, LayeredSpace, NumpyPatchGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import numpy as np

from .agents import SsAgent


class SugarPatchGrid(NumpyPatchGrid):
    def __init__(self, max_values, **kwargs):
        super().__init__(**kwargs)
        self.max_values = max_values

    def step(self):
        super().step()
        self._grid += 1
        self._grid[self._grid > self.max_values] = self.max_values[self._grid > self.max_values]

class SugarscapeCg(Model):
    '''
    Sugarscape 2 Constant Growback
    '''

    verbose = True  # Print-monitoring

    def __init__(self, height=50, width=50,
                 initial_population=100):
        '''
        Create a new Constant Growback model with the given parameters.

        Args:
            initial_population: Number of population to start with
        '''

        # Set parameters
        self.height = height
        self.width = width
        self.initial_population = initial_population

        self.schedule = RandomActivation(self)

        sugar_max_distribution = np.genfromtxt("sugarscape_cg/sugar-map.txt")
        sugar_patches = SugarPatchGrid(sugar_max_distribution, torus=False)
        self.grid = LayeredSpace({"agents": Grid(height=self.height, width=self.width, torus=False),
                                "sugar": sugar_patches})
        self.schedule.add(sugar_patches)

        self.datacollector = DataCollector({"SsAgent": lambda m: m.schedule.get_agent_count(), })

        # Create agent:
        for i in range(self.initial_population):
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            sugar = self.random.randrange(6, 25)
            metabolism = self.random.randrange(2, 4)
            vision = self.random.randrange(1, 6)
            ssa = SsAgent((x, y), self, False, sugar, metabolism, vision)
            self.grid.place_agent(ssa, (x, y))
            self.schedule.add(ssa)

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)
        if self.verbose:
            print([self.schedule.time,
                   self.schedule.get_breed_count(SsAgent)])

    def run_model(self, step_count=200):

        if self.verbose:
            print('Initial number Sugarscape Agent: ',
                  self.schedule.get_breed_count(SsAgent))

        for i in range(step_count):
            self.step()

        if self.verbose:
            print('')
            print('Final number Sugarscape Agent: ',
                  self.schedule.get_breed_count(SsAgent))
