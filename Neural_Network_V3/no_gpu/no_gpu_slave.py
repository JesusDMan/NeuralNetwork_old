import os
import pickle, socket
import threading
import time
import numpy as np
import colorama

colorama.init()
print(colorama.Fore.GREEN, end='')

with open(os.getcwd() + '/globals.txt') as f: globals = f.read().split('\n')
print(globals)

HOST_IP = globals[0]
GENERAL_PORT = int(globals[1])
UPDATER_PORT = int(globals[2])
SLAVES_PORT = int(globals[3])
SPLITTER = bytes(globals[4], 'UTF8')
BIAS_MOMENTUM = WEIGHT_MOMENTUM = None

GENERAL_COMMU = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM, proto=0)
GENERAL_COMMU.connect((HOST_IP, GENERAL_PORT))
UPDATER_COMMU = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM, proto=0)
UPDATER_COMMU.connect((HOST_IP, UPDATER_PORT))

LAYERS = []
INPUTS = OUTPUTS = []


# ~~~~ UTILS ~~~~

def raise_exception(err): print(colorama.Fore.RED + err + colorama.Fore.GREEN)


# Receivers:
def receive_net() -> None:
    """
    Receives the net (layer by layer) to the LAYERS variable, through the GENERAL socket.
    :return: None.
    """
    while True:
        if GENERAL_COMMU.recv(3) != b'nxt':  # Can be 'end' or something else (because of an error)
            break
        activation_func = str(GENERAL_COMMU.recv(int.from_bytes(GENERAL_COMMU.recv(1), 'big')), 'UTF8')
        print(activation_func)
        l = GENERAL_COMMU.recv(int.from_bytes(GENERAL_COMMU.recv(5), 'big'), socket.MSG_WAITALL).split(SPLITTER)
        LAYERS.append((pickle.loads(l[0]), pickle.loads(l[1]), find_activation_function(activation_func)))


def receive_input() -> None:
    """
    Receives an INPUTS batch and TRUE OUTPUTS batch to the INPUTS, OUTPUTS variables, through the GENERAL socket.
    :return: None.
    """
    global INPUTS, OUTPUTS
    INPUTS = pickle.loads(GENERAL_COMMU.recv(int.from_bytes(GENERAL_COMMU.recv(8), 'big')))
    OUTPUTS = pickle.loads(GENERAL_COMMU.recv(int.from_bytes(GENERAL_COMMU.recv(8), 'big')))


def receive_param_idx(SLAVE_COMMU: socket) -> tuple:
    """
    Receives a net-parameter's place (indexes) in the net, through a SLAVE socket.
    :param SLAVE_COMMU: A socket for communication with each slave.
    :return: The parameter's place in the Neural Network.
    """
    PARAM = pickle.loads(SLAVE_COMMU.recv(int.from_bytes(SLAVE_COMMU.recv(1), 'big')))
    return PARAM


# Run net:
def run_net(inp: np.ndarray) -> np.ndarray:
    """
    Runs the net on a given input.
    :param inp: The net's input.
    :return: The net's output.
    """
    for l in LAYERS:
        inp = run_layer(l, inp)
    return inp


def run_layer(l: np.ndarray, inp: np.ndarray) -> np.ndarray:
    """
    Runs a specific layer on a given input (with matrix multiplication).
    :param l: The layer (weights and biases matrix).
    :param inp: The layer's input.
    :return: The layer's output.
    """
    return np.apply_along_axis(arr=(l[1].dot(inp) + l[0]), func1d=l[2], axis=0)


# TRAINING:
def change(PARAM: tuple) -> bytes:
    """
    Checks what change in a given parameter in the net improves its performance.
    Changes are: Increasing, decreasing, and not-changing.
    :return: 2 for increase, 0 for decrease, 1 for not-changing ~~ in bytes.
    """
    m1 = mistake_value(INPUTS, OUTPUTS)

    if PARAM[0] == 'B':
        LAYERS[PARAM[1]][0][PARAM[2]] += BIAS_MOMENTUM
    else:
        LAYERS[PARAM[1]][1][PARAM[2]][PARAM[3]] += WEIGHT_MOMENTUM
    m2 = mistake_value(INPUTS, OUTPUTS)

    if PARAM[0] == 'B':
        LAYERS[PARAM[1]][0][PARAM[2]] -= 2 * BIAS_MOMENTUM
    else:
        LAYERS[PARAM[1]][1][PARAM[2]][PARAM[3]] -= 2 * WEIGHT_MOMENTUM
    m0 = mistake_value(INPUTS, OUTPUTS)

    if PARAM[0] == 'B':
        LAYERS[PARAM[1]][0][PARAM[2]] += BIAS_MOMENTUM
    else:
        LAYERS[PARAM[1]][1][PARAM[2]][PARAM[3]] += WEIGHT_MOMENTUM

    l = [m0, m1, m2]
    return int.to_bytes(l.index(min(l)), 1, 'big')


def mistake_value(inputs: list, outputs: list) -> float:
    """
    Calculates the net's mistake value (for a batch).
    :param inputs: The inputs batch
    :param outputs: The true-outputs batch
    :return: The mistake value
    """
    mistake_sum = 0
    for idx in range(len(inputs)):
        real_output = outputs[idx]
        net_output = run_net(inputs[idx])
        mistake_sum += np.sum(np.apply_along_axis(arr=(net_output - real_output), func1d=np.abs, axis=0))
    return mistake_sum / len(inputs)


def update_net() -> None:
    """
    Updates the net - The function is a thread that gets parameter-indexes in the net and a change needed to be done,
    through the UPDATER socket, and changes it.
    :return: None.
    """
    while True:
        change = int.from_bytes(UPDATER_COMMU.recv(1), 'big')
        change -= 1  # if change == 0: change = -1, if change == 2: change = 1
        param_idx = UPDATER_COMMU.recv(int.from_bytes(UPDATER_COMMU.recv(1), 'big'))
        param_idx = pickle.loads(param_idx)
        if param_idx[0] == 'B':
            LAYERS[param_idx[1]][0][param_idx[2]] += BIAS_MOMENTUM * change
        else:
            LAYERS[param_idx[1]][1][param_idx[2]][param_idx[3]] += WEIGHT_MOMENTUM * change


# ======================================================================================================================
# Activation functions

def find_activation_function(activation_function):
    if activation_function == 'strong sigmoid': return strong_sigmoid
    if activation_function == 'weak sigmoid': return weak_sigmoid
    if activation_function == 'weaker sigmoid': return weaker_sigmoid
    if activation_function == 'weaker sigmoidX2': return weaker_sigmoidX2
    if activation_function == 'tanh': return tanh
    if activation_function == 'weak tanh': return weak_tanh
    if activation_function == 'weaker tanh': return weaker_tanh
    if activation_function == 'reLu': return reLu
    return reg  # Default (activation_function may be 'weaker sigmoid, doesn't matter)


def reg(x): return x


def strong_sigmoid(x): return 1 / (1 + np.e ** (-x))


def weak_sigmoid(x): return 2 / (1 + np.e ** (-0.5 * x)) - 1  # 2 / (1 + np.e ** x) - 1


def weaker_sigmoid(x): return 2 / (1 + 0.9 ** x) - 1  # 2 / (1 + np.e ** (-0.1 * x)) - 1  #


def weaker_sigmoidX2(x): return 4 / (1 + 0.9 ** x) - 2  # 2 / (1 + np.e ** (-0.1 * x)) - 1  #


def tanh(x): return np.tanh(x)


def weak_tanh(x): return np.tanh(0.25 * x)


def weaker_tanh(x): return np.tanh(0.1 * x)


def reLu(x): return np.maximum(-1, x)


# ======================================================================================================================
# MAINS


def slave():
    SLAVE_COMMU = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM, proto=0)
    SLAVE_COMMU.connect((HOST_IP, SLAVES_PORT))
    while not INPUTS:
        pass
    while True:
        PARAM = receive_param_idx(SLAVE_COMMU)
        should_change = change(PARAM)
        param: bytes = pickle.dumps(PARAM)
        SLAVE_COMMU.send(bytes(should_change) + int.to_bytes(len(param), 4, 'big') + param)


def wait_for_server_cmd():
    global BIAS_MOMENTUM, WEIGHT_MOMENTUM
    while True:
        msg = GENERAL_COMMU.recv(3)
        if msg == b'INP':  # New input
            receive_input()
            print(INPUTS[0])

        elif msg == b'ADD':  # Add one slave
            threading.Thread(target=slave, daemon=False).start()

        elif msg == b'CBM':  # Bias Momentum
            BIAS_MOMENTUM = pickle.loads(GENERAL_COMMU.recv(21))
        elif msg == b'CWM':  # Weight Momentum
            WEIGHT_MOMENTUM = pickle.loads(GENERAL_COMMU.recv(21))
        elif msg == b'CAM':  # Both
            BIAS_MOMENTUM = WEIGHT_MOMENTUM = pickle.loads(GENERAL_COMMU.recv(21))


if __name__ == '__main__':
    print('Starting...')
    receive_net()
    GENERAL_COMMU.recv(3)
    receive_input()
    BIAS_MOMENTUM = pickle.loads(GENERAL_COMMU.recv(21))
    WEIGHT_MOMENTUM = pickle.loads(GENERAL_COMMU.recv(21))
    print(BIAS_MOMENTUM, WEIGHT_MOMENTUM)

    number_of_slaves = int(input('Enter shit: '))
    # number_of_slaves = int.frombytes(GENERAL_COMMU.recv(2), 'big')

    GENERAL_COMMU.send(int.to_bytes(number_of_slaves, 2, 'big'))
    threading.Thread(target=update_net, daemon=False).start()
    for i in range(number_of_slaves):
        threading.Thread(target=slave, daemon=False).start()

    threading.Thread(target=wait_for_server_cmd, daemon=False).start()

    while True:
        cmd = input('Enter CMD: ')

        if cmd == 'print moment':
            print(f'Bias Momentum: {BIAS_MOMENTUM}, Weight Momentum: {WEIGHT_MOMENTUM}')

