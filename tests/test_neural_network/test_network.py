import unittest
import numpy as np

import src
import src.neural_network.network as network
import numpy as np

'''
EXAMPLE_RECIPE = {'inpLayer': [{'label': 'A1', 'activFunc': 'idem', 'bias': 0.0, 'idx': 'inp1'},
                               {'label': 'A2', 'activFunc': 'idem', 'bias': 0.0, 'idx': 'inp2'}],
                  'hidLayer': [{'label': 'ha1', 'activFunc': 'idem', 'bias': 0.0, 'idx': 'hid1'},
                               {'label': 'ha2', 'activFunc': 'idem', 'bias': 0.0, 'idx': 'hid2'}],
                  'outLayer': [{'label': 'A1', 'activFunc': 'idem', 'bias': 0.0, 'idx': 'out1'},
                               {'label': 'A2', 'activFunc': 'idem', 'bias': 0.0, 'idx': 'out2'}],
                  'inpToHidConnections': [{'fromNeuron': 'inp1', 'toNeuron': 'hid1', 'weight': 0.1},
                                          {'fromNeuron': 'inp1', 'toNeuron': 'hid2', 'weight': 0.2},
                                          {'fromNeuron': 'inp2', 'toNeuron': 'hid1', 'weight': 0.3},
                                          {'fromNeuron': 'inp2', 'toNeuron': 'hid2', 'weight': 0.4}],
                  'hidToOutConnections': [{'fromNeuron': 'hid1', 'toNeuron': 'out1', 'weight': 0.1},
                                          {'fromNeuron': 'hid1', 'toNeuron': 'out2', 'weight': 0.2},
                                          {'fromNeuron': 'hid2', 'toNeuron': 'out1', 'weight': 0.3},
                                          {'fromNeuron': 'hid2', 'toNeuron': 'out2', 'weight': 0.4}],
                  'recConnections': [{'fromNeuron': 'out1', 'toNeuron': 'inp1', 'weight': 0.2},
                                     {'fromNeuron': 'out2', 'toNeuron': 'inp2', 'weight': 0.4}], 'nn_factors': []}
'''


class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_digits(self):
        recipe = src.logic.EXAMPLE_RECIPE

        lp = src.logic.LogicProgram.from_dict(recipe['lp'])
        ag = src.logic.Clause.from_dict(recipe['abductive_goal'])
        factors = src.logic.Factors.from_dict(recipe['factors'])

        nn_recipe = src.connect.get_nn_recipe(lp, ag, factors)
        nn = network.NeuralNetwork3L.from_dict(nn_recipe)

        nn.train(np.array([[-1, -1, -1, -1]]), np.array([[1, 1, 1]]), 100, on_stabilised=True)

        self.assertIsInstance(nn.forward(np.array([-1, -1, -1, -1])), np.ndarray)


if __name__ == '__main__':
    unittest.main()
