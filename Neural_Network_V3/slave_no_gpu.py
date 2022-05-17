input('start')
try:
    from os import getcwd
    from pickle import loads, dumps
    from socket import socket, AF_INET, SOCK_STREAM, MSG_WAITALL
    from threading import Thread
    import numpy as np
    # from torch import Tensor, sum, abs, sigmoid, tanh, relu
    # from torch.cuda import device_count
    from colorama import Fore
    import colorama
    colorama.init()

    # SETUP SHIT
    print(Fore.GREEN, end='')

    with open(getcwd() + '\\globals.txt') as f: global_param = f.read().split('\n')
    print(global_param)

    HOST_IP = global_param[0]
    GENERAL_PORT = int(global_param[1])
    UPDATER_PORT = int(global_param[2])
    SLAVES_PORT = int(global_param[3])
    SPLITTER = bytes(global_param[4], 'UTF8')
    BIAS_MOMENTUM = WEIGHT_MOMENTUM = None



    GENERAL_COMMU = socket(family=AF_INET, type=SOCK_STREAM, proto=0)
    GENERAL_COMMU.connect((HOST_IP, GENERAL_PORT))
    UPDATER_COMMU = socket(family=AF_INET, type=SOCK_STREAM, proto=0)
    UPDATER_COMMU.connect((HOST_IP, UPDATER_PORT))

    LAYERS = []
    INPUTS = OUTPUTS = []
except Exception as e:
    input('Miau'+str(e))

# ~~~~ UTILS ~~~~

def raise_exception(err): print(Fore.RED + err + Fore.GREEN)


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
        layer = GENERAL_COMMU.recv(int.from_bytes(GENERAL_COMMU.recv(5), 'big'), MSG_WAITALL).split(SPLITTER)
        layer = (loads(layer[0]).numpy().squeeze(), loads(layer[1]).numpy().squeeze(),
                 find_activation_function(activation_func))
        LAYERS.append(layer)


def receive_input() -> None:
    """
    Receives an INPUTS batch and TRUE OUTPUTS batch to the INPUTS, OUTPUTS variables, through the GENERAL socket.
    :return: None.
    """
    global INPUTS, OUTPUTS
    ff = loads(GENERAL_COMMU.recv(int.from_bytes(GENERAL_COMMU.recv(8), 'big')))

    INPUTS = [t.numpy().squeeze() for t in ff]

    ff = loads(GENERAL_COMMU.recv(int.from_bytes(GENERAL_COMMU.recv(8), 'big')))
    OUTPUTS = [t.numpy().squeeze() for t in ff]
    # print(INPUTS[0])
    # print(INPUTS[0].transpose(-1, -2))


def receive_param_idx(slave_commu: socket) -> tuple:
    """
    Receives a net-parameter's place (indexes) in the net, through a SLAVE socket.
    :param slave_commu: A socket for communication with each slave.
    :return: The parameter's place in the Neural Network.
    """
    PARAM = loads(slave_commu.recv(int.from_bytes(slave_commu.recv(1), 'big')))
    return PARAM


# Run net:
def run_net(inp):
    """
    Runs the net on a given input.
    :param inp: The net's input.
    :return: The net's output.
    """
    for layer in LAYERS:
        # print(inp)
        inp = np.apply_along_axis(arr=(layer[1].dot(inp) + layer[0]), func1d=layer[2], axis=0)
    return inp


# def run_layer(layer, inp):
#     """
#     Runs a specific layer on a given input (with matrix multiplication).
#     :param layer: The layer (weights and biases matrix).
#     :param inp: The layer's input.
#     :return: The layer's output.
#     """
#     return layer[1].mm(inp) + layer[0]


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

def mms(x):
    sum = 0
    for i in x:
        if i<0:
            sum +=(-i*5)
        else:
            sum += i
    return sum
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
        true_idx = 0
        mistake_sum += sum(abs(net_output - real_output)) + abs(net_output[true_idx] - real_output[true_idx]) * 5    # mistake_sum += np.sum(np.apply_along_axis(arr=(net_output - real_output), func1d=np.abs, axis=0))
    return mistake_sum


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
        param_idx = loads(param_idx)
        # print(param_idx)
        if param_idx[0] == 'B':
            LAYERS[param_idx[1]][0][param_idx[2]] += BIAS_MOMENTUM * change
        else:
            LAYERS[param_idx[1]][1][param_idx[2]][param_idx[3]] += WEIGHT_MOMENTUM * change


# ======================================================================================================================
# Activation functions

def find_activation_function(activation_function):
    if activation_function == 'sigmoid': return np.tanh
    if activation_function == 'tanh': return np.tanh
    if activation_function == 'reLu': return relu

def relu(x):
    return np.maximum(-1,x)


# ======================================================================================================================
# MAINS

def slave():
    SLAVE_COMMU = socket(family=AF_INET, type=SOCK_STREAM, proto=0)
    SLAVE_COMMU.connect((HOST_IP, SLAVES_PORT))

    while True:
        PARAM = receive_param_idx(SLAVE_COMMU)
        should_change = change(PARAM)
        param: bytes = dumps(PARAM)
        SLAVE_COMMU.send(bytes(should_change) + int.to_bytes(len(param), 4, 'big') + param)


def wait_for_server_cmd():
    global BIAS_MOMENTUM, WEIGHT_MOMENTUM
    while True:
        msg = GENERAL_COMMU.recv(3)
        if msg == b'INP':  # New input
            receive_input()

        elif msg == b'ADD':  # Add one slave
            Thread(target=slave, daemon=False).start()

        elif msg == b'CBM':  # Bias Momentum
            BIAS_MOMENTUM = loads(GENERAL_COMMU.recv(21))
        elif msg == b'CWM':  # Weight Momentum
            WEIGHT_MOMENTUM = loads(GENERAL_COMMU.recv(21))
        elif msg == b'CAM':  # Both
            BIAS_MOMENTUM = WEIGHT_MOMENTUM = loads(GENERAL_COMMU.recv(21))


if __name__ == '__main__':
    print('Starting...')
    receive_net()
    GENERAL_COMMU.recv(3)
    receive_input()
    BIAS_MOMENTUM = loads(GENERAL_COMMU.recv(21))
    WEIGHT_MOMENTUM = loads(GENERAL_COMMU.recv(21))
    print(BIAS_MOMENTUM, WEIGHT_MOMENTUM)
    number_of_slaves = int(input('Enter shit: '))

    # number_of_slaves = int.frombytes(GENERAL_COMMU.recv(2), 'big')
    GENERAL_COMMU.send(int.to_bytes(number_of_slaves, 2, 'big'))

    run_net(INPUTS[0])
    Thread(target=update_net, daemon=False).start()
    for i in range(number_of_slaves):
        Thread(target=slave, daemon=False).start()

    Thread(target=wait_for_server_cmd, daemon=False).start()
