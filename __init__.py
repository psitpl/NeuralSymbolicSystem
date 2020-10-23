import src

recipe = src.logic.EXAMPLE_RECIPE

lp = src.logic.LogicProgram.from_dict(recipe['lp'])
ag = src.logic.Clause.from_dict(recipe['abductive_goal'])
factors = src.logic.Factors.from_dict(recipe['factors'])

import src.neural_network.network as network
import numpy as np

nn_recipe = src.connect.get_nn_recipe(lp, ag, factors)
nn = network.NeuralNetwork3L.from_dict(nn_recipe)

x = np.array([-1, -1, -1, -1, 1])
y = np.array([1, 1, 1, 1, 1])