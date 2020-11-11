import json
import itertools
from typing import List, Iterable

import matplotlib.pyplot as plt
import numpy as np

import src.neural_network.activations

inp_layer_example = [{"label": "A1", "activFunc": "sigm", "bias": 0.0, "idx": "inp1"},
                     {"label": "A2", "activFunc": "sigm", "bias": 0.0, "idx": "inp2"},
                     {"label": "A3", "activFunc": "sigm", "bias": 0.0, "idx": "inp3"}]

hid_layer_example = [{"label": "ha1", "activFunc": "sigm", "bias": 0.0, "idx": "hid1"},
                     {"label": "ha2", "activFunc": "sigm", "bias": 0.0, "idx": "hid2"}]

out_layer_example = [{"label": "A1", "activFunc": "sigm", "bias": 0.0, "idx": "out1"},
                     {"label": "A2", "activFunc": "sigm", "bias": 0.0, "idx": "out2"}]

connec_i2h_sample = [{'fromNeuron': 'inp1', 'toNeuron': 'hid1', 'weight': 0.1},
                     {'fromNeuron': 'inp1', 'toNeuron': 'hid2', 'weight': 0.2},
                     {'fromNeuron': 'inp2', 'toNeuron': 'hid1', 'weight': 0.3},
                     {'fromNeuron': 'inp2', 'toNeuron': 'hid2', 'weight': 0.4},
                     {'fromNeuron': 'inp3', 'toNeuron': 'hid1', 'weight': 0.3},
                     {'fromNeuron': 'inp3', 'toNeuron': 'hid2', 'weight': 0.4}]

connec_h2o_sample = [{'fromNeuron': 'hid1', 'toNeuron': 'out1', 'weight': 0.1},
                     {'fromNeuron': 'hid1', 'toNeuron': 'out2', 'weight': 0.2},
                     {'fromNeuron': 'hid2', 'toNeuron': 'out1', 'weight': 0.3},
                     {'fromNeuron': 'hid2', 'toNeuron': 'out2', 'weight': 0.4}]

connec_o2i_sample = [{'fromNeuron': 'out1', 'toNeuron': 'inp1', 'weight': 0.2},
                     {'fromNeuron': 'out2', 'toNeuron': 'inp2', 'weight': 0.4}]

architecture_example = {'inpLayer': inp_layer_example,
                        'hidLayer': hid_layer_example,
                        'outLayer': out_layer_example,
                        'inpToHidConnections': connec_i2h_sample,
                        'hidToOutConnections': connec_h2o_sample,
                        'recConnections': connec_o2i_sample,
                        'nn_factors': []}


def mean_squarred_error(y, output):
    return (y - output) ** 2


def d_mean_squarred_error(y, output):
    return 2 * (output - y)


def valuation(x, a_min):
    if (type(x) == list) or (type(x) == np.ndarray):
        return [valuation(elem, a_min) for elem in x]
    else:
        if x <= (-1 * a_min):
            return -1
        elif x >= a_min:
            return 1
        return 0


def act_f(f: str) -> callable:
    """
    :param f: string name of function
    :return: fuction of taht name from activations.py
    """
    if f == "idem":
        raw_f = src.neural_network.activations.idem
    elif f == "const":
        raw_f = src.neural_network.activations.const
    elif f == "sigm" or f == "tanh":
        raw_f = src.neural_network.activations.sigm
    else:
        raise ValueError(f"There is no function named {f} in activation function list.")

    def wrapped_f(x: float) -> float:
        return raw_f(min(1.0, max(-1.0, x)))

    setattr(wrapped_f, "d", raw_f.__getattribute__("d"))

    return wrapped_f


def to_binary(matrix: np.ndarray) -> np.ndarray:
    """
    Creates boolean mask of given array.

    :param matrix: numpy.ndarray
    :return: numpy.ndarray

    """
    return np.vectorize(bool)(matrix)


def all_combinations(l: int) -> itertools.product:
    return itertools.product([-1, 1], repeat=l)


class LayerInfo:

    def __init__(self, specification: List[dict]):
        self.specification = specification

        self.label = [neuron['label'] for neuron in specification]
        self.f = [neuron['activFunc'] for neuron in specification]
        self.bias = [neuron['bias'] for neuron in specification]
        self.idx = [neuron['idx'] for neuron in specification]

        self.len = len(specification)


def flatten_rec_layer(connections: [dict]) -> [dict]:
    flat_connections = []
    rec_layer_froms = dict()
    rec_layer_tos = dict()

    for connection in connections:

        if connection['toNeuron'].startswith('rec'):
            # Create mapping {"out3": "recA2"} // {"outLayer": "addRecLayer"}
            rec_layer_froms[connection['fromNeuron']] = connection['toNeuron']

        elif connection['fromNeuron'].startswith('rec'):
            # Create mapping {"recA2": "inp3"} // {"addRecLayer": "inpLayer"}
            rec_layer_tos[connection['fromNeuron']] = connection['toNeuron']

        else:
            flat_connections.append(connection)

    for from_neuron, rec_neuron in rec_layer_froms.items():
        to_neuron = rec_layer_tos[rec_neuron]
        flat_connections.append({'fromNeuron': from_neuron, 'toNeuron': to_neuron, 'weight': 1.0})

    return flat_connections


def set_weights(connections: [dict], prev_layer: LayerInfo, next_layer: LayerInfo) -> np.ndarray:
    """
    Sets weights based on connections dicts and layers.

    :param connections: [dict]
    :param prev_layer: LayerInfo
    :param next_layer: LayerInfo
    :return: numpy.ndarray

    """
    weights = np.zeros((next_layer.len, prev_layer.len))

    prev_dict = dict(zip(prev_layer.idx, [i for i in range(prev_layer.len)]))
    next_dict = dict(zip(next_layer.idx, [i for i in range(next_layer.len)]))

    for connection in connections:
        weights[next_dict[connection['toNeuron']]][prev_dict[connection['fromNeuron']]] = connection['weight']

    return weights


def vectorize_with_report(f: callable) -> callable:

    def try_(try_f: callable, *try_args, **try_kwargs):
        try:
            return try_f(*try_args, **try_kwargs)
        except Exception as e:
            print("Values when error was raised:")
            print("args:", try_args)
            print("kwargs:", try_kwargs)
            raise e

    def vectorized(x: Iterable, *args, **kwargs):
        return [try_(f, elem, *args, **kwargs) for elem in x]

    return vectorized


def get_model(values: [int], order: [str]) -> dict:
    model = {"positive": [], "negative": []}
    for value, label in zip(values, order):
        if value == 1:
            model["positive"].append(label)
        elif value == -1:
            model["negative"].append(label)
    return model


class NeuralNetwork3L:

    def __init__(self, architecture: dict, factors: src.logic.Factors, eta=0.01):

        self.architecture = architecture

        self.inp_layer_spec = LayerInfo(architecture['inpLayer'])
        self.hid_layer_spec = LayerInfo(architecture['hidLayer'])
        self.out_layer_spec = LayerInfo(architecture['outLayer'])

        self.i2h_connections = set_weights(architecture['inpToHidConnections'],
                                           self.inp_layer_spec, self.hid_layer_spec)
        self.h2o_connections = set_weights(architecture['hidToOutConnections'],
                                           self.hid_layer_spec, self.out_layer_spec)
        self.o2i_connections = set_weights(flatten_rec_layer(architecture['recConnections']),
                                           self.out_layer_spec, self.inp_layer_spec)

        self.factors = factors
        self.eta = eta

        # TODO: Make sure that truth neuron can't switch weight
        self.i2h_b = to_binary(self.i2h_connections)
        self.h2o_b = to_binary(self.h2o_connections)
        self.o2i_b = to_binary(self.o2i_connections)

        self.inp_layer_aggregated = np.zeros(self.inp_layer_spec.len)
        self.hid_layer_aggregated = np.zeros(self.hid_layer_spec.len)
        self.out_layer_aggregated = np.zeros(self.out_layer_spec.len)

        self.inp_layer_calculated = np.zeros(self.inp_layer_spec.len)
        self.hid_layer_calculated = np.zeros(self.hid_layer_spec.len)
        self.out_layer_calculated = np.zeros(self.out_layer_spec.len)

        self.inp_layer_activation = vectorize_with_report(act_f(self.inp_layer_spec.f[0]))
        self.hid_layer_activation = vectorize_with_report(act_f(self.hid_layer_spec.f[0]))
        self.out_layer_activation = vectorize_with_report(act_f(self.out_layer_spec.f[0]))

        self.inp_layer_activation_derivation = vectorize_with_report(act_f(self.inp_layer_spec.f[0]).__getattribute__('d'))
        self.hid_layer_activation_derivation = vectorize_with_report(act_f(self.hid_layer_spec.f[0]).__getattribute__('d'))
        self.out_layer_activation_derivation = vectorize_with_report(act_f(self.out_layer_spec.f[0]).__getattribute__('d'))

        self.errors = []

        self.output_error = None
        self.output_delta = None
        self.hidden_error = None
        self.hidden_delta = None

        self.shape = f"{self.inp_layer_spec.len}x{self.hid_layer_spec.len}x{self.out_layer_spec.len}"

    @staticmethod
    def from_dict(d: dict):
        return NeuralNetwork3L(architecture=d['nn'], factors=src.logic.Factors(**d['nnFactors']))

    @staticmethod
    def from_json(j: str):
        return NeuralNetwork3L.from_dict(json.loads(j))

    @staticmethod
    def from_file(j: str):
        with open(j, 'r') as json_file:
            return NeuralNetwork3L.from_dict(json.load(json_file))

    @staticmethod
    def from_lp(lp: src.logic.LogicProgram, ag: src.logic.Clause, factors: src.logic.Factors):

        nn_recipe = src.connect.get_nn_recipe(lp, ag, factors)
        return NeuralNetwork3L.from_dict(nn_recipe)

    @staticmethod
    def from_dropped(fp: str):
        with open(fp, 'r') as json_file:
            dropped = json.load(json_file)

        nn = NeuralNetwork3L.from_dict(dropped)
        nn.i2h_connections = dropped['i2h_connections']
        nn.h2o_connections = dropped['h2o_connections']
        nn.o2i_connections = dropped['o2i_connections']

        nn.i2h_b = to_binary(nn.i2h_connections)
        nn.h2o_b = to_binary(nn.h2o_connections)
        nn.o2i_b = to_binary(nn.o2i_connections)

        return nn

    def _pack(self) -> dict:
        return {'architecture': self.architecture,
                'i2h_connections': self.i2h_connections.tolist(),
                'h2o_connections': self.h2o_connections.tolist(),
                'o2i_connections': self.o2i_connections.tolist(),
                'factors': self.factors.to_dict()}

    def drop(self, fp: str):
        d = self._pack()
        with open(fp, 'w') as json_file:
            json.dump(d, json_file)

    def to_lp(self):

        return src.connect.get_lp_from_nn(order_inp=[{"idx": idx, "label": label} for idx, label in zip(self.inp_layer_spec.idx, self.inp_layer_spec.label)],
                                          order_out=[{"idx": idx, "label": label} for idx, label in zip(self.out_layer_spec.idx, self.out_layer_spec.label)],
                                          amin=self.factors.amin,
                                          io_pairs=self.get_io_pairs())

    def draw(self, fig=None, ax=None, save: str = '',
             left: float = .1, right: float = .9, bottom: float = .1, top: float = .9):
        """
        Draw a neural network cartoon using matplotilb.

        :parameters:
            - ax : matplotlib.axes.AxesSubplot
                The axes on which to plot the cartoon (get e.g. by plt.gca())
            - left : float
                The center of the leftmost node(s) will be placed here
            - right : float
                The center of the rightmost node(s) will be placed here
            - bottom : float
                The center of the bottommost node(s) will be placed here
            - top : float
                The center of the topmost node(s) will be placed here
        """

        if fig is None:
            fig = plt.figure(figsize=(12, 12))
        if ax is None:
            ax = fig.gca()
            ax.axis('off')

        layer_sizes = [self.inp_layer_spec.len, self.hid_layer_spec.len,
                       self.out_layer_spec.len, self.inp_layer_spec.len]
        v_spacing = (top - bottom) / float(max(layer_sizes))
        h_spacing = (right - left) / 3.0

        # Nodes
        layers = [self.inp_layer_spec, self.hid_layer_spec, self.out_layer_spec, self.inp_layer_spec]
        for n, layer_size in enumerate(layer_sizes):
            layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.
            for m in range(layer_size):
                circle_coordinates = (n * h_spacing + left, layer_top - m * v_spacing)
                circle = plt.Circle(circle_coordinates, v_spacing / 4.,
                                    color='w', ec='k', zorder=4)
                ax.add_artist(circle)
                ax.annotate(layers[n].label[m], xy=circle_coordinates, fontsize=15, ha="center", zorder=100)

        # Edges
        connections = [self.i2h_connections, self.h2o_connections, self.o2i_connections]
        for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            layer_top_a = v_spacing * (layer_size_a - 1) / 2. + (top + bottom) / 2.
            layer_top_b = v_spacing * (layer_size_b - 1) / 2. + (top + bottom) / 2.
            for m in range(layer_size_a):
                for o in range(layer_size_b):
                    if connections[n][o][m]:
                        line = plt.Line2D([n * h_spacing + left, (n + 1) * h_spacing + left],
                                          [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing], c='k',
                                          linewidth=connections[n][o][m])
                        ax.add_artist(line)

        if save:
            fig.savefig(save)

        return fig

    def plot_errors(self):
        plt.plot(self.errors)
        plt.show()

    def calculate_input(self, x: np.ndarray = None):
        pass

    def forward(self, x: np.ndarray):
        """
        Implementation of feed forward.

        Function that calculates output of Neural Network by given input.
        :param: input_vector: one-dimensional numpy.ndarray or other iterable
                one dimensional object.
        """

        self.inp_layer_aggregated = x - self.inp_layer_spec.bias
        self.inp_layer_calculated = np.array(self.inp_layer_activation(self.inp_layer_aggregated))

        self.hid_layer_aggregated = self.i2h_connections.dot(self.inp_layer_aggregated) - self.hid_layer_spec.bias
        self.hid_layer_calculated = np.array(self.hid_layer_activation(self.hid_layer_aggregated))

        self.out_layer_aggregated = self.h2o_connections.dot(self.hid_layer_aggregated) - self.out_layer_spec.bias
        self.out_layer_calculated = np.array(self.out_layer_activation(self.out_layer_aggregated))

        return self.out_layer_calculated

    def backprop(self, y: np.ndarray, eta: float):
        """
        Implementation of Backpropagation algorithm.
        Function that calculates the error on the output layer, propagates it
        through network and modifies weights according using delta rule.

        Every weight in network is modified using following equation:

        ΔW_ho = η * (y-o) * f'(v) * h   , where
        η     = learning constant, usually very small, around 0.01 (self.eta)
        (y-o) = squared error derivative
        y     = values of the output layer (self.XXX_layer_calculated)
        o     = values expected on the output (x)
        f'    = derivative of the activation function (self.XXX_layer_activation_derivation)
        v     = aggregated values on next layer

        More about Backpropagation agorithm:
        > https://en.wikipedia.org/wiki/Backpropagation

        More about Delta Rule:
        > https://en.wikipedia.org/wiki/Delta_rule

        """
        self.output_error = d_mean_squarred_error(y, self.out_layer_calculated)  # error in output
        self.output_delta = self.output_error * self.out_layer_activation_derivation(self.out_layer_aggregated)

        # z2 error: how much our hidden layer weights contribute to output error
        self.hidden_error = self.output_delta.dot(self.h2o_connections)

        # applying derivative of sigmoid to z2 error
        self.hidden_delta = self.hidden_error * self.hid_layer_activation_derivation(self.hid_layer_aggregated)

        # adjusting first set (input -> hidden) weights
        l2h_weights_delta = np.outer(self.inp_layer_calculated.T, self.hidden_delta).T * eta
        self.i2h_connections += l2h_weights_delta * self.i2h_b

        # adjusting second set (hidden -> output) weights
        h2o_weights_delta = np.outer(self.hid_layer_calculated.T, self.output_delta).T * eta
        self.h2o_connections += h2o_weights_delta * self.h2o_b

        print(l2h_weights_delta)
        print(h2o_weights_delta)

        return sum(self.output_error) / len(self.output_error)

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int, on_stabilised: bool = False,
              stop_when: callable = lambda e: e == 0, vis=False):

        examples_n = len(x)
        self.errors = []

        fig = None

        for epoch in range(epochs):
            for i, (x_, y_) in enumerate(zip(x, y)):
                if on_stabilised:
                    self.stabilize(x_)
                else:
                    self.forward(x_)
                avg_error = self.backprop(y_, self.eta)
                # self.update_weights(new_i2h_connections, new_h2o_connections, eta)
                self.errors.append(avg_error)
                print(f'Epoch {epoch + 1}/{epochs} | Example {i + 1}/{examples_n} | Error: {avg_error}')

                if vis:
                    if fig is not None:
                        fig.clear()
                    fig = self.draw(fig=fig)

                if stop_when(avg_error):
                    return 0

    def stabilize(self, x: Iterable = None):
        """
        All inputs to -1

        One neuron always return True.

        Network is considered stable when input and output (True - None - False)

        """

        if x is None:
            x = np.array([-1 for _ in range(self.inp_layer_spec.len)])
        if len(x) != self.inp_layer_spec.len:
            raise ValueError(f"x must have length {self.inp_layer_spec.len}, has {len(x)} instead.")
        x[-1] = 1

        last_output_vector = []

        output_vector = self.forward(x)
        tp_iteration = 0

        print("Tp Operator iteration:", tp_iteration)
        print("Output vector:", valuation(output_vector, self.factors.amin))
        print("Model", get_model(valuation(output_vector, self.factors.amin), self.out_layer_spec.label))

        while valuation(last_output_vector, self.factors.amin) != valuation(output_vector, self.factors.amin):

            # TODO: Find where is bug
            last_output_vector = output_vector
            input_vector = self.o2i_connections.dot(self.out_layer_aggregated)
            input_vector[-1] = 1
            output_vector = self.forward(input_vector)

            tp_iteration += 1

            print("Tp Operator iteration:", tp_iteration)
            print("Output vector:", valuation(output_vector, self.factors.amin))
            print("Model", get_model(valuation(output_vector, self.factors.amin), self.out_layer_spec.label))

        return output_vector

    def get_io_pairs(self):
        inputs = all_combinations(self.inp_layer_spec.len)
        io_pairs = []

        for x in inputs:
            y = self.forward(np.array(list(x)))
            io_pairs.append((list(x), y.tolist()))

        return io_pairs





