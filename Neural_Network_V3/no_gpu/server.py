try:
    import os
    import pickle, socket, time, random, colorama, sys, os.path as path, numpy as np
    from threading import Thread
    from NeuralNetwork_no_gpu import utils
    from NeuralNetwork_no_gpu.neural_network import NeuralNetwork
    from NeuralNetwork_no_gpu.batch import Batch
except Exception as e:
    input(e)

# GLOBAL PARAMETERS
MAIN_PORT = 420
UPDATER_PORT = 666
SLAVES_PORT = 6969
HOST_IP = socket.gethostbyname(socket.gethostname())

# USER PARAMETERS
DATASET_SIZE = None
DATASET_LOCATION = None
IN_BATCHES = None
SAVE_NET = None

SLAVES = []
test_batch = Batch(200, 2)
test_batch.images = [np.array([5, -9]), np.array([-6, 6]), np.array([1, -5]), np.array([-9, -4]), np.array([4, 5]),
                     np.array([-7, 1]), np.array([1, 0]), np.array([-7, -4]), np.array([-9, -6]), np.array([4, 8]),
                     np.array([0, -4]), np.array([0, 2]), np.array([-3, -9]), np.array([1, 7]), np.array([-4, 2]),
                     np.array([-3, -8]), np.array([-6, -8]), np.array([0, -4]), np.array([1, -5]), np.array([4, -3]),
                     np.array([2, 7]), np.array([-9, 4]), np.array([-7, 2]), np.array([-1, 9]), np.array([0, -6]),
                     np.array([3, -6]), np.array([-6, -8]), np.array([4, -2]), np.array([8, -6]), np.array([5, -1]),
                     np.array([-2, 6]), np.array([4, -5]), np.array([-1, -8]), np.array([-4, 5]), np.array([3, 7]),
                     np.array([-8, 8]), np.array([4, -7]), np.array([-3, 4]), np.array([6, 3]), np.array([0, -3]),
                     np.array([-1, 5]), np.array([1, 5]), np.array([3, -7]), np.array([-5, -7]), np.array([8, 5]),
                     np.array([-5, -1]), np.array([-3, 9]), np.array([0, 5]), np.array([3, -2]), np.array([1, 0]),
                     np.array([4, 6]), np.array([-3, 0]), np.array([-8, 2]), np.array([-8, -4]), np.array([-1, 7]),
                     np.array([-7, -3]), np.array([9, 4]), np.array([-5, 1]), np.array([-3, 2]), np.array([-6, 7]),
                     np.array([6, 3]), np.array([-9, -4]), np.array([-1, -8]), np.array([3, -2]), np.array([-8, 7]),
                     np.array([3, -3]), np.array([-2, -8]), np.array([5, 0]), np.array([5, -7]), np.array([-1, 4]),
                     np.array([0, 6]), np.array([9, 8]), np.array([-2, 2]), np.array([-5, -4]), np.array([-9, 0]),
                     np.array([7, 4]), np.array([-3, 2]), np.array([9, 7]), np.array([6, 4]), np.array([-7, -4]),
                     np.array([-3, 2]), np.array([2, -2]), np.array([-9, 6]), np.array([0, -6]), np.array([9, -6]),
                     np.array([-6, -7]), np.array([-2, 0]), np.array([2, -3]), np.array([-3, 4]), np.array([6, -9]),
                     np.array([-9, 5]), np.array([0, -8]), np.array([2, -1]), np.array([7, -2]), np.array([-4, 5]),
                     np.array([-5, -6]), np.array([1, 9]), np.array([-8, 6]), np.array([0, 8]), np.array([4, 6]),
                     np.array([1, -9]), np.array([6, 7]), np.array([-4, -3]), np.array([-6, 3]), np.array([4, 1]),
                     np.array([-4, -1]), np.array([-4, 5]), np.array([-7, 8]), np.array([4, -1]), np.array([1, -1]),
                     np.array([3, 0]), np.array([0, -6]), np.array([-7, 9]), np.array([8, -6]), np.array([7, 9]),
                     np.array([-2, 9]), np.array([8, -4]), np.array([-7, -2]), np.array([9, 0]), np.array([-4, 1]),
                     np.array([-8, -3]), np.array([-1, -4]), np.array([-4, 3]), np.array([-8, -5]), np.array([5, -4]),
                     np.array([-7, -6]), np.array([6, 4]), np.array([-7, 8]), np.array([8, -7]), np.array([3, -5]),
                     np.array([-1, 0]), np.array([-3, 6]), np.array([4, 7]), np.array([5, -4]), np.array([-9, -3]),
                     np.array([-8, -3]), np.array([3, 4]), np.array([3, -3]), np.array([-5, 3]), np.array([-1, 8]),
                     np.array([4, -4]), np.array([-6, 3]), np.array([4, -6]), np.array([-4, -1]), np.array([-2, -9]),
                     np.array([-8, 4]), np.array([-7, 5]), np.array([7, -9]), np.array([4, 0]), np.array([-1, -7]),
                     np.array([-4, 2]), np.array([3, 7]), np.array([9, -5]), np.array([1, -4]), np.array([-2, 3]),
                     np.array([0, -6]), np.array([-6, -8]), np.array([-4, 3]), np.array([-9, 4]), np.array([-5, -2]),
                     np.array([-5, -4]), np.array([-2, 7]), np.array([4, 5]), np.array([5, -2]), np.array([0, -7]),
                     np.array([0, -7]), np.array([7, -1]), np.array([8, -4]), np.array([6, -8]), np.array([-6, 4]),
                     np.array([-9, -2]), np.array([7, 5]), np.array([9, 3]), np.array([4, -4]), np.array([-8, 8]),
                     np.array([9, 3]), np.array([2, 8]), np.array([8, 9]), np.array([8, 9]), np.array([7, 6]),
                     np.array([2, -1]), np.array([0, -9]), np.array([5, -1]), np.array([0, 4]), np.array([0, -8]),
                     np.array([9, 2]), np.array([9, -7]), np.array([0, 3]), np.array([-9, -5]), np.array([0, -8]),
                     np.array([6, 9]), np.array([4, -9]), np.array([-6, -3]), np.array([4, -8]), np.array([0, 6]),
                     np.array([-6, 0]), np.array([-8, 1]), np.array([8, 7]), np.array([6, -3]), np.array([2, 4])]
test_batch.labels = [np.array([1, 0]), np.array([0, 1]), np.array([1, 0]), np.array([0, 1]), np.array([0, 1]),
                     np.array([0, 1]), np.array([1, 0]), np.array([0, 1]), np.array([0, 1]), np.array([0, 1]),
                     np.array([1, 0]), np.array([0, 1]), np.array([1, 0]), np.array([0, 1]), np.array([0, 1]),
                     np.array([1, 0]), np.array([1, 0]), np.array([1, 0]), np.array([1, 0]), np.array([1, 0]),
                     np.array([0, 1]), np.array([0, 1]), np.array([0, 1]), np.array([0, 1]), np.array([1, 0]),
                     np.array([1, 0]), np.array([1, 0]), np.array([1, 0]), np.array([1, 0]), np.array([1, 0]),
                     np.array([0, 1]), np.array([1, 0]), np.array([1, 0]), np.array([0, 1]), np.array([0, 1]),
                     np.array([0, 1]), np.array([1, 0]), np.array([0, 1]), np.array([1, 0]), np.array([1, 0]),
                     np.array([0, 1]), np.array([0, 1]), np.array([1, 0]), np.array([1, 0]), np.array([1, 0]),
                     np.array([0, 1]), np.array([0, 1]), np.array([0, 1]), np.array([1, 0]), np.array([1, 0]),
                     np.array([0, 1]), np.array([0, 1]), np.array([0, 1]), np.array([0, 1]), np.array([0, 1]),
                     np.array([0, 1]), np.array([1, 0]), np.array([0, 1]), np.array([0, 1]), np.array([0, 1]),
                     np.array([1, 0]), np.array([0, 1]), np.array([1, 0]), np.array([1, 0]), np.array([0, 1]),
                     np.array([1, 0]), np.array([1, 0]), np.array([1, 0]), np.array([1, 0]), np.array([0, 1]),
                     np.array([0, 1]), np.array([1, 0]), np.array([0, 1]), np.array([0, 1]), np.array([0, 1]),
                     np.array([1, 0]), np.array([0, 1]), np.array([1, 0]), np.array([1, 0]), np.array([0, 1]),
                     np.array([0, 1]), np.array([1, 0]), np.array([0, 1]), np.array([1, 0]), np.array([1, 0]),
                     np.array([1, 0]), np.array([0, 1]), np.array([1, 0]), np.array([0, 1]), np.array([1, 0]),
                     np.array([0, 1]), np.array([1, 0]), np.array([1, 0]), np.array([1, 0]), np.array([0, 1]),
                     np.array([1, 0]), np.array([0, 1]), np.array([0, 1]), np.array([0, 1]), np.array([0, 1]),
                     np.array([1, 0]), np.array([0, 1]), np.array([0, 1]), np.array([0, 1]), np.array([1, 0]),
                     np.array([0, 1]), np.array([0, 1]), np.array([0, 1]), np.array([1, 0]), np.array([1, 0]),
                     np.array([1, 0]), np.array([1, 0]), np.array([0, 1]), np.array([1, 0]), np.array([0, 1]),
                     np.array([0, 1]), np.array([1, 0]), np.array([0, 1]), np.array([1, 0]), np.array([0, 1]),
                     np.array([0, 1]), np.array([1, 0]), np.array([0, 1]), np.array([0, 1]), np.array([1, 0]),
                     np.array([0, 1]), np.array([1, 0]), np.array([0, 1]), np.array([1, 0]), np.array([1, 0]),
                     np.array([0, 1]), np.array([0, 1]), np.array([0, 1]), np.array([1, 0]), np.array([0, 1]),
                     np.array([0, 1]), np.array([0, 1]), np.array([1, 0]), np.array([0, 1]), np.array([0, 1]),
                     np.array([1, 0]), np.array([0, 1]), np.array([1, 0]), np.array([0, 1]), np.array([1, 0]),
                     np.array([0, 1]), np.array([0, 1]), np.array([1, 0]), np.array([1, 0]), np.array([1, 0]),
                     np.array([0, 1]), np.array([0, 1]), np.array([1, 0]), np.array([1, 0]), np.array([0, 1]),
                     np.array([1, 0]), np.array([1, 0]), np.array([0, 1]), np.array([0, 1]), np.array([0, 1]),
                     np.array([0, 1]), np.array([0, 1]), np.array([0, 1]), np.array([1, 0]), np.array([1, 0]),
                     np.array([1, 0]), np.array([1, 0]), np.array([1, 0]), np.array([1, 0]), np.array([0, 1]),
                     np.array([0, 1]), np.array([1, 0]), np.array([1, 0]), np.array([1, 0]), np.array([0, 1]),
                     np.array([1, 0]), np.array([0, 1]), np.array([0, 1]), np.array([0, 1]), np.array([1, 0]),
                     np.array([1, 0]), np.array([1, 0]), np.array([1, 0]), np.array([0, 1]), np.array([1, 0]),
                     np.array([1, 0]), np.array([1, 0]), np.array([0, 1]), np.array([0, 1]), np.array([1, 0]),
                     np.array([0, 1]), np.array([1, 0]), np.array([0, 1]), np.array([1, 0]), np.array([0, 1]),
                     np.array([0, 1]), np.array([0, 1]), np.array([1, 0]), np.array([1, 0]), np.array([0, 1])]
train_d = utils.build_dataset(dataset_size=1000, batch_size=30, input_size=2)
current_batch = train_d[0]
INPUTS = current_batch.images
OUTPUTS = current_batch.labels


net = NeuralNetwork(name='NN_5L_MAIN', bias_momentum=0.05, weight_momentum=0.05,
                    train_dataset=train_d, test_batch=test_batch)

net.add_layer(2, 'weak sigmoid')
net.add_layer(500, 'weak sigmoid')
net.add_layer(500, 'weak sigmoid')
net.add_layer(500, 'weak sigmoid')
net.add_layer(500, 'weak sigmoid')
net.add_layer(2, 'weak sigmoid')
# net = neural_network.restore_net(os.getcwd(), 'NN_5L_MAIN')
net.test_batch = test_batch
net.train_dataset = train_d

globals_ = '\n'.join(
    [str(x) for x in
     [HOST_IP, MAIN_PORT, UPDATER_PORT, SLAVES_PORT, '~~~~~~~~~']])
with open(path.split(sys.argv[0])[0] + '/globals.txt', 'w') as f: f.write(globals_)

MAIN_COMMU = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM, proto=0)
MAIN_COMMU.bind((HOST_IP, MAIN_PORT))
MAIN_COMMU.listen(5)

UPDATER_COMMU = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM, proto=0)
UPDATER_COMMU.bind((HOST_IP, UPDATER_PORT))
UPDATER_COMMU.listen(5)

SLAVES_COMMU = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM, proto=0)
SLAVES_COMMU.bind((HOST_IP, SLAVES_PORT))
SLAVES_COMMU.listen(10)

colorama.init()
print(colorama.Fore.GREEN, end='')


# UTILS

def raise_exception(err):
    print(colorama.Fore.RED + err + colorama.Fore.GREEN)


def chain_slaves(slaves_manager: list, number_of_slaves: int) -> None:
    """
    Every 'slave manager' has many slaves (threads) that need to be added.
    :param slaves_manager: A component in SLAVES that contains a slaves list.
    :param number_of_slaves: The amount of slaves for this slave manager.
    :return: None
    """
    for _ in range(number_of_slaves):
        slaves_manager[2].append([SLAVES_COMMU.accept()[0], False])


def chain_slave_managers() -> None:
    """
    Adds new 'slave managers' (computers) to the training network.
    Every slave manager has slaves (threads) that actually work in the training process, so it's basically:
    SLAVES[slave_manager[slave_1, slave_2...]...]
    :return: None.
    """
    while True:
        slaves_manager = [MAIN_COMMU.accept()[0], UPDATER_COMMU.accept()[0], []]
        try:
            send_net(slaves_manager[0])
            send_input(slaves_manager, INPUTS, OUTPUTS)
            slaves_manager[0].send(pickle.dumps(net._bias_momentum))
            slaves_manager[0].send(pickle.dumps(net._weight_momentum))
            number_of_slaves = int.from_bytes(slaves_manager[0].recv(2), 'big')
            # number_of_slaves = slaves_manager[0].send(int.to_bytes(30, 2, 'big'))
            Thread(target=chain_slaves, args=(slaves_manager, number_of_slaves), name='duck').start()
            SLAVES.append(slaves_manager)
        except:
            if slaves_manager in SLAVES:
                SLAVES.remove(slaves_manager)


# SLAVE MANAGEMENT

def find_free_slave() -> int:
    """
    Searches for an available slave
    :return: The id of an available slave
    """
    while True:
        if SLAVES:
            for slaves_manager in SLAVES:
                for slave in slaves_manager[2]:
                    if slave[1] == False:
                        return slave


def kill_slave(slave: list) -> None:
    """
    This function gets the [socket, bool] object and deletes it in SLAVES
    :param slave:
    :return:
    """
    for slaves_manager in SLAVES:
        if slave in slaves_manager[2]:
            try:
                slave_manager[0].send(b'')
                if slave in slaves_manager[2]: slaves_manager[2].remove(slave)
            except:
                try:
                    SLAVES.remove(slaves_manager)
                    raise_exception('Fuck a slave manager got away')
                except:
                    pass
            break


def number_of_slvs():
    s = 0
    for slave_manager in SLAVES:
        s += len(slave_manager[2])
    return s


def use_slave(slave: list, parameter_idx: tuple) -> None:
    """
    Send a computer the net, input, output, and parameter's indexes, and get the results.
    :param parameter_idx: The indexes of the parameter that will be changed.
    :return: None
    """
    slave[1] = True
    client: socket = slave[0]
    try:
        send_parameter_idx(client, parameter_idx)
    except:
        kill_slave(slave)
        exit()

    get_feedback(slave)


def task_manager() -> None:
    """
    Goes through the network's parameters and creating work for the slaves.
    :return: None.i[i
    """
    global current_batch, INPUTS, OUTPUTS
    net.__mistakes_log__ = [utils.batch_mistake_value(net=net, batch=test_batch)]
    for i in range(10000):

        for slave_manager in SLAVES:
            send_input(slave_manager, INPUTS, OUTPUTS)
        start = time.time()
        for layer in reversed(net.layers):
            for bias_idx in range(layer.size):
                Thread(target=use_slave,
                       args=(find_free_slave(), ('B', layer.index, bias_idx))).start()
            for weight_idx_I in range(layer.size):
                for weight_idx_II in range(layer.prev_size):
                    Thread(target=use_slave,
                           args=(find_free_slave(), ('W', layer.index, weight_idx_I, weight_idx_II))).start()

        ms = utils.batch_mistake_value(net=net, batch=test_batch)
        net.__mistakes_log__.append(ms)
        if min(net.__mistakes_log__) == ms:
            net.save()
        net.graph()
        with open(path.join(os.getcwd(), f'{net._name}_log.txt'), 'a') as f:
            f.write(f'Time = {time.time() - start}, i = {i} Input index: {0} '
                    f'Input: {INPUTS[0]} Output: {OUTPUTS[0]}, NO: {net.run(INPUTS[0])}, MV: {ms}, SLVS: {number_of_slvs()}\n')
        current_batch = net.train_dataset[random.randint(0, len(net.train_dataset) - 1)]
        INPUTS = current_batch.images
        OUTPUTS = current_batch.labels


# SEND NETWORK'S PARAMETERS


def send_net(slave_socket: socket) -> None:
    """
    Send the net to a slave.
    :return: None
    """

    for layer in net.layers:
        b_layer = bytes(layer)
        slave_socket.send(b'nxt')  # There is another layer
        activation_func = bytes(layer.activation_function_name, 'UTF8')
        slave_socket.send(int.to_bytes(len(activation_func), 1, 'big') + activation_func)
        slave_socket.send(int.to_bytes(len(b_layer), 5, 'big') + b_layer)
    slave_socket.send(b'end')


def send_parameter_idx(slave_socket: socket, param_idx: tuple) -> None:
    """
    Send the parameter indexes to a slave. Parameter's indexes shape: [TYPE, layerIdx, paramIdx/sLayerIdx...]
    :param slave_socket: The slave's socket.
    :param param_idx: The parameter's indexes.
    :return: None.
    """

    param_idx: bytes = pickle.dumps(param_idx)
    slave_socket.send(int.to_bytes(len(param_idx), 1, 'big') + param_idx)


def send_input(slaves_manager: list, input: np.array, output: np.array) -> None:
    """
    Send a slave the input and true output.
    :param slaves_manager: The slave manager to which the data will be sent.
    :param input: The input.
    :param output: The true output.
    :return: None.
    """

    manager_socket = slaves_manager[0]
    try:
        manager_socket.send(b'INP')
        inp = pickle.dumps(input)
        out = pickle.dumps(output)
        manager_socket.send(int.to_bytes(len(inp), 8, 'big') + inp)
        manager_socket.send(int.to_bytes(len(out), 8, 'big') + out)
    except:
        try:
            manager_socket.send(b'')
        except:
            if manager_socket in SLAVES: SLAVES.remove(manager_socket)


def get_feedback(slave: list) -> None:
    """
    Get the feedback back from the slave (if the parameter should be changed or not), then activates 'fix_net'.
    :return: None.
    """
    slave_socket: socket = slave[0]
    try:
        change: int = int.from_bytes(slave_socket.recv(1), 'big')  # 0 -> decrease, 1 -> stay, 2 -> increase
        param_idx: tuple = pickle.loads(slave_socket.recv(int.from_bytes(slave_socket.recv(4), 'big')))
        print(param_idx)
    except:
        try:
            slave_socket.send(b'')
        except:
            kill_slave(slave)
            exit()
        change = 1

    if change != 1:  # There is a change needed to be applied
        fix_net(change, param_idx)
        Thread(target=update_slaves_nets, args=(param_idx, change)).start()
    slave[1] = False


def update_slaves_nets(param_idx: tuple, change):
    slave_manager = None
    try:
        change = int.to_bytes(change, 1, 'big')
        param_idx: bytes = pickle.dumps(param_idx)

        for slave_manager in SLAVES:
            slave_manager[1].send(change + int.to_bytes(len(param_idx), 1, 'big') + param_idx)
    except:
        if slave_manager in SLAVES:
            SLAVES.remove(slave_manager)
        raise_exception('F')


def fix_net(should_change: int, parameter_idx: tuple) -> None:
    """
    Change the parameter if it improves performance, and give the slave new task.
    :param should_change: True if changing the parameter is positive, False otherwise.
    :param parameter_idx: The parameter's indexes.
    :return: None
    """
    layer_idx = parameter_idx[1]
    if parameter_idx[0] == 'B':
        if should_change == 0:
            net.layers[layer_idx].biases[parameter_idx[2]] -= net._bias_momentum
        else:
            net.layers[layer_idx].biases[parameter_idx[2]] += net._bias_momentum


    else:  # For Weights
        if should_change == 0:
            net.layers[layer_idx].weights[parameter_idx[2]][parameter_idx[3]] -= net._weight_momentum
        else:
            net.layers[layer_idx].weights[parameter_idx[2]][parameter_idx[3]] += net._weight_momentum


# MAIN
print('Starting...')
Thread(target=chain_slave_managers, name='chainer', daemon=False).start()
Thread(target=task_manager, name='task manager', daemon=False).start()

while True:
    cmd = input('Enter CMD: ')
    try:
        print(colorama.Fore.RED, end='')
        if 'add' in cmd:
            amount = int(cmd.strip('add '))
            for _ in range(amount):
                for slave_manager in SLAVES:
                    slave_manager[0].send(b'ADD')
                    chain_slaves(slave_manager, 1)

        elif 'cbm' in cmd:
            new_bm = float(cmd.strip('cbm = '))
            net.change_momentum('bias', new_bm)
            for slave_manager in SLAVES:
                slave_manager[0].send(b'CBM' + pickle.dumps(new_bm))

        elif 'cwm' in cmd:
            new_wm = float(cmd.strip('cwm = '))
            net.change_momentum('weight', new_wm)
            for slave_manager in SLAVES:
                slave_manager[0].send(b'CWM' + pickle.dumps(new_wm))

        elif 'cam' in cmd:
            new_m = float(cmd.strip('cam = '))
            net.change_momentum('bias', new_m)
            net.change_momentum('weight', new_m)
            new_m = pickle.dumps(new_m)
            for slave_manager in SLAVES:
                slave_manager[0].send(b'CAM' + new_m)

        elif cmd == 'save net':
            net.save()

        elif cmd == 'example':
            idx = random.randint(0, len(INPUTS) - 1)
            print(f'Inp: {INPUTS[idx]}, Out: {OUTPUTS[idx]}, Net: {net.run(INPUTS[idx])}')


        elif cmd == 'print moment':
            print(f'Bias Momentum: {net._bias_momentum}, Weight Momentum: {net._weight_momentum}')

        elif cmd == 'print net':
            print(str(net))



        elif cmd == 'print mistake':
            print(utils.batch_mistake_value(net, test_batch))

        elif cmd == 'exit':
            break

        else:
            print(f'\'{cmd}\' ~ illegal command')

    except:
        print(f'\'{cmd}\' ~ illegal command')
    print(colorama.Fore.GREEN, end='')
