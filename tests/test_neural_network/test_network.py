import unittest
import numpy as np

import src.neural_network.network as network

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


class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_digits(self):
        nn = network.NeuralNetwork3L(network.architecture_example)
        nn.forward(np.array([0.5, 0.2]))
        self.assertAlmostEqual(0.065, nn.out_layer_calculated[0])
        self.assertAlmostEqual(0.094, nn.out_layer_calculated[1])


if __name__ == '__main__':
    unittest.main()
