import os
import pickle, socket, time, random, colorama, sys, os.path as path
from threading import Thread
from NeuralNetwork import neural_network, utils
from NeuralNetwork.neural_network import NeuralNetwork
from NeuralNetwork.batch import Batch

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
test_batch = Batch(50, 784)
train_d = utils.build_dataset(dataset_size=50, batch_size=5, input_size=784)
current_batch = train_d[0]
INPUTS = current_batch.images
OUTPUTS = current_batch.labels

net = NeuralNetwork(name='NN_5L_MAIN_GGG', bias_momentum=0.05, weight_momentum=0.05,
                    train_dataset=train_d, test_batch=test_batch)

net.add_layer(784, 'sigmoid')
net.add_layer(200, 'sigmoid')
# net.add_layer(300, 'sigmoid')
# net.add_layer(300, 'sigmoid')
net.add_layer(10, 'reLu')

# utils.calc_progress(('B', 0,0), net)
# utils.calc_progress(('B', 1,3), net)
# utils.calc_progress(('W', 0,0,0), net)
# utils.calc_progress(('W', 0,0,1), net)
# net = neural_network.restore_net(os.getcwd(), 'NN_5L_MAIN')
# net.test_batch = test_batch
# net.train_dataset = train_d
# print(net.run(test_batch.images[0]))

mm = 0
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

current_time = 0
current_param = ()
update_param_flag = False


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
    global current_batch, current_time, mm, current_param, INPUTS, OUTPUTS
    net.__mistakes_log__ = [utils.batch_mistake_value(net=net, batch=test_batch)]
    for i in range(10000):
        for slave_manager in SLAVES:
            send_input(slave_manager, INPUTS, OUTPUTS)
        start = time.time()
        current_time = time.time()
        update_param_flag = True
        mm = 0
        for layer in net.layers:
            print(f'Layer: idx={layer.index}, size={layer.size}, time={time.time() - start}')
            for bias_idx in range(layer.size):
                if update_param_flag:
                    current_param = ('B', layer.index, bias_idx)
                mm += 1
                if mm == 1 or mm == 2 or mm == 615442 or mm == 10000:
                    print(current_param)
                Thread(target=use_slave,
                       args=(find_free_slave(), ('B', layer.index, bias_idx))).start()
            for weight_idx_I in range(layer.size):
                for weight_idx_II in range(layer.prev_size):
                    if update_param_flag:
                        current_param = ('W', layer.index, weight_idx_I, weight_idx_II)
                    mm += 1
                    if mm == 1 or mm == 2 or mm == 615442 or mm == 10000:
                        print(current_param)
                    Thread(target=use_slave,
                           args=(find_free_slave(), ('W', layer.index, weight_idx_I, weight_idx_II))).start()

        net.lets_be_loggin_shit(time=(time.time() - start), batch_index=i, number_of_slaves=number_of_slvs())
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


def send_input(slaves_manager: list, input, output) -> None:
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
if __name__ == '__main__':
    print('Starting...')
    Thread(target=chain_slave_managers, name='chainer', daemon=False).start()
    Thread(target=task_manager, name='task manager', daemon=False).start()
    ss = time.time()
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
                Bidx = random.randint(0, len(net.train_dataset) - 1)
                idx = random.randint(0, net.train_dataset[Bidx].size - 1)
                print(f'Out: {net.train_dataset[Bidx].labels[idx]}, Net: {net.run(net.train_dataset[Bidx].images[idx])}, MS: {utils.batch_mistake_value(net, net.train_dataset[Bidx])}Path: {net.train_dataset[Bidx].paths[idx]}')

            elif cmd == 'dataset stat':
                print(f'Training Dataset size: {len(net.train_dataset)}\n'
                      f'Training Batch size: {net.train_dataset[0].size}\n'
                      f'Testing Batch size: {net.test_batch.size}\n'
                      f'Input size (TB): {net.test_batch.input_size}\n'
                      f'Input size (TDB): {net.train_dataset[0].input_size}')

            elif cmd == 'print moment':
                print(f'Bias Momentum: {net._bias_momentum}, Weight Momentum: {net._weight_momentum}')

            elif cmd == 'print net':
                print(str(net))

            elif cmd == 'print full net':
                print(repr(net))

            elif cmd == 'print param':
                update_param_flag = True
                time.sleep(0.1)
                t = time.time()
                print(current_param, t-ss, mm)
                update_param_flag = False
                ss = t

            elif cmd == 'print mistake':
                print(utils.batch_mistake_value(net, test_batch))

            elif cmd == 'print time':
                print(time.time() - current_time)

            elif cmd == 'exit':
                break

            else:
                print(f'\'{cmd}\' ~ illegal command')

        except:
            print(f'\'{cmd}\' ~ illegal command')
        print(colorama.Fore.GREEN, end='')
