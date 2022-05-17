# ============================================== *** INITIALIZE *** ==============================================
try:
    # ~~~ IMPORTS
    from os import getcwd
    from pickle import loads, dumps
    from socket import socket, AF_INET, SOCK_STREAM, MSG_WAITALL
    from threading import Thread
    from torch import Tensor
    from colorama import Fore
    from torch.nn import SiLU, Softmax, Softmin, Sigmoid, ReLU, ReLU6, LeakyReLU, Tanh
except Exception as e:
    input(e)

# ~~~ SETUP SHIT

print(Fore.GREEN, end='')
with open(getcwd() + '\\globals.txt') as f:
    global_param = f.read().split('\n')
print(global_param)

# ~~ General parameters ~~
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

# Global
LAYERS = []
INPUTS = OUTPUTS = []
mistake_tuple = (-1,0,1)

# ============================================== *** UTILS *** ==============================================

def raise_exception(err): print(Fore.RED + err + Fore.GREEN)


def find_activation_function(activation_function):
    if activation_function == 'sigmoid': return Sigmoid()
    if activation_function == 'chill sigmoid': return lambda x: Sigmoid()(x/70)
    if activation_function == 'SiLU': return SiLU()
    if activation_function == 'chill SiLU': return lambda x: SiLU()(x*0.1)
    if activation_function == 'reLu': return ReLU()
    if activation_function == 'softmax': return Softmax(dim=0)
    if activation_function == 'softmax': return Softmin(dim=0)
    if activation_function == 'reLu6': return ReLU6()
    if activation_function == 'leaky reLu': return LeakyReLU()
    if activation_function == 'tanh': return Tanh()
    if activation_function == 'pass': return lambda x: x


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ *** RECEIVERS *** ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

        layer = (Tensor(loads(layer[0])), Tensor(loads(layer[1])), find_activation_function(activation_func))
        LAYERS.append(layer)


def receive_input() -> None:
    """
    Receives an INPUTS batch and TRUE OUTPUTS batch to the INPUTS, OUTPUTS variables, through the GENERAL socket.
    :return: None.
    """
    global INPUTS, OUTPUTS
    INPUTS = [inp for inp in loads(GENERAL_COMMU.recv(int.from_bytes(GENERAL_COMMU.recv(8), 'big')))]
    OUTPUTS = [inp for inp in loads(GENERAL_COMMU.recv(int.from_bytes(GENERAL_COMMU.recv(8), 'big')))]


def receive_param_idx(slave_commu: socket) -> tuple:
    """
    Receives a net-parameter's place (indexes) in the net, through a SLAVE socket.
    :param slave_commu: A socket for communication with each slave.
    :return: The parameter's place in the Neural Network.
    """
    PARAM = loads(slave_commu.recv(int.from_bytes(slave_commu.recv(1), 'big')))
    return PARAM


# ============================================== * MAIN * ==============================================

def run_net(inp: Tensor) -> Tensor:
    """
    Runs the net on a given input.
    :param inp: The net's input.
    :return: The net's output.
    """
    inp = LAYERS[0][2](LAYERS[0][1]*inp + LAYERS[0][0])
    for layer in LAYERS[1:]:
        inp = layer[2](layer[1].mm(inp) + layer[0])
    return inp


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ *** TRAINING *** ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def change(PARAM: tuple) -> Tensor:
    """
    Checks what change in a given parameter in the net improves its performance.
    Changes are: Increasing, decreasing, and not-changing.
    :return: 2 for increase, 0 for decrease, 1 for not-changing ~~ in bytes.
    """
    m0 = mistake_value(INPUTS, OUTPUTS)
    p0 = PARAM[0]  # Parameter type
    p1 = PARAM[1]  # Layer index
    p2 = PARAM[2]  # Bias index/second layer index

    if p0 == 'B':
        LAYERS[p1][0][p2] += BIAS_MOMENTUM
        m1 = mistake_value(INPUTS, OUTPUTS)
        LAYERS[p1][0][p2] -= 2*BIAS_MOMENTUM
        m2 = mistake_value(INPUTS, OUTPUTS)
        LAYERS[p1][0][p2] += BIAS_MOMENTUM
        l = (m2, m0, m1)
        return LAYERS[p1][0][p2] + (l.index(min(l))-1)*WEIGHT_MOMENTUM
    else:
        LAYERS[p1][1][p2][PARAM[3]] += WEIGHT_MOMENTUM
        m1 = mistake_value(INPUTS, OUTPUTS)
        LAYERS[p1][1][p2][PARAM[3]] -= 2*WEIGHT_MOMENTUM
        m2 = mistake_value(INPUTS, OUTPUTS)
        LAYERS[p1][1][p2][PARAM[3]] += WEIGHT_MOMENTUM
        l = (m2, m0, m1)
        print(p0, l.index(min(l))-1)
        return LAYERS[p1][1][p2][PARAM[3]] + (l.index(min(l))-1)*WEIGHT_MOMENTUM



def mistake_value(inputs: list, outputs: list) -> Tensor:
    """
    Calculates the net's mistake value (for a batch).
    :param inputs: The inputs batch
    :param outputs: The true-outputs batch
    :return: The mistake value
    """
    mistake_sum = Tensor([[0]])

    for i in range(len(inputs)):
        mistake_sum += run_net(inputs[i]) - outputs[i]
    return mistake_sum.squeeze(1)/len(outputs)


# ============================================== * MAIN FUNCTIONS * ==============================================

def slave():
    SLAVE_COMMU = socket(family=AF_INET, type=SOCK_STREAM, proto=0)
    SLAVE_COMMU.connect((HOST_IP, SLAVES_PORT))

    while True:
        PARAM = receive_param_idx(SLAVE_COMMU)
        new_val_b: bytes = dumps(change(PARAM))
        param_b: bytes = dumps(PARAM)
        SLAVE_COMMU.send(int.to_bytes(len(new_val_b), 3, 'big') + new_val_b + int.to_bytes(len(param_b), 4, 'big') + param_b)


def update_net() -> None:
    """
    Updates the net - The function is a thread that gets parameter-indexes in the net and a change needed to be done,
    through the UPDATER socket, and changes it.
    :return: None.
    """
    while True:
        new_val = loads(UPDATER_COMMU.recv(int.from_bytes(UPDATER_COMMU.recv(3), 'big')))
        param_idx = loads(UPDATER_COMMU.recv(int.from_bytes(UPDATER_COMMU.recv(1), 'big')))
        if param_idx[0] == 'B':
            LAYERS[param_idx[1]][0][param_idx[2]] = new_val
        else:
            LAYERS[param_idx[1]][1][param_idx[2]][param_idx[3]] = new_val

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


# =====================================================================================================================

if __name__ == '__main__':
    print('Starting...')
    receive_net()
    GENERAL_COMMU.recv(3)
    receive_input()
    BIAS_MOMENTUM = loads(GENERAL_COMMU.recv(21))
    WEIGHT_MOMENTUM = loads(GENERAL_COMMU.recv(21))
    print(BIAS_MOMENTUM, WEIGHT_MOMENTUM)
    number_of_slaves = input('Enter the number of slaves: ')
    while not number_of_slaves.isnumeric():
        number_of_slaves = input('A number, please... Enter again: ')

    number_of_slaves = int(number_of_slaves)

    GENERAL_COMMU.send(int.to_bytes(number_of_slaves, 2, 'big'))

    Thread(target=update_net, daemon=False).start()
    Thread(target=wait_for_server_cmd, daemon=False).start()

    for i in range(number_of_slaves):
        Thread(target=slave, daemon=False).start()