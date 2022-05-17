# ============================================== *** INITIALIZE *** ==============================================
# ~~~ IMPORTS
import os, pickle, socket, time, random, colorama, sys, os.path as path
from threading import Thread
from NeuralNetwork import neural_network, utils
from NeuralNetwork.neural_network import NeuralNetwork
from NeuralNetwork.batch import Batch
from torch import Tensor

colorama.init()
print(colorama.Fore.GREEN, end='')

# ~~~ BUILD NN ~~~
# Dataset configuration:
test_batch: Batch = utils.build_test_batch(batch_size=200, input_size=11,
                                           data_location=r'C:\Desktop\archive\train_oSwQCTC (1)\train.fact.csv')  # A batch for testing net's performance
train_d: list = utils.build_sales_pred_dataset(batch_size=200, input_size=11)

# ~~ New net: ~~
net = NeuralNetwork(name='NN_5L_MAIN', bias_momentum=0.01, weight_momentum=0.01, train_dataset=train_d,
                    test_batch=test_batch)
net.add_layer(11, 'SiLU')
net.add_layer(30, 'chill sigmoid')
net.add_layer(50, 'chill sigmoid')
net.add_layer(40, 'chill sigmoid')
net.add_layer(30, 'softmax')
net.add_layer(1, 'leaky reLu')

# ~~ Restore net: ~~
# net = neural_network.restore_net(os.getcwd(), 'NN_5L_MAIN')  # Restore the net from a file
# net.test_batch = test_batch
# net.train_dataset = train_d

# ~~~ NETWORK CONFIGURATION ~~~
# ~~ General parameters ~~
MAIN_PORT: int = 420
SELF_IP: str = socket.gethostbyname(socket.gethostname())
UPDATER_PORT: int = 666
SLAVES_PORT: int = 6969

# ~~ Sockets configuration ~~
MAIN_COMMU: socket.socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM, proto=0)
MAIN_COMMU.bind((SELF_IP, MAIN_PORT))
MAIN_COMMU.listen(5)

UPDATER_COMMU: socket.socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM, proto=0)
UPDATER_COMMU.bind((SELF_IP, UPDATER_PORT))
UPDATER_COMMU.listen(5)

SLAVES_COMMU: socket.socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM, proto=0)
SLAVES_COMMU.bind((SELF_IP, SLAVES_PORT))
SLAVES_COMMU.listen(10)

# Updating the 'globals' file (that's used by the slave)
globals_ = '\n'.join([str(x) for x in [SELF_IP, MAIN_PORT, UPDATER_PORT, SLAVES_PORT, '~~~~~~~~~']])
with open(path.split(sys.argv[0])[0] + '/globals.txt', 'w') as f: f.write(globals_)

# ----- GLOBAL SHIT -----
SLAVES: list = []  # The list of slave managers
current_batch: Batch = train_d[random.randint(0, len(train_d) - 1)]  # The current batch that the net trains on
INPUTS: list = current_batch.images  # The current batch's inputs images
OUTPUTS: list = current_batch.labels  # The current batch's output labels
progress: int = 0  # The number of parameters the net has gone through
current_param: tuple = ()  # The current parameter the net's training /
# /(updates only when 'update_param_flag' is on.
update_param_flag: bool = False  # When on, the 'current_param' is being updated. Off most of the time.


# ============================================== *** UTILS *** ==============================================

def raise_exception(err: str) -> None:
    """
    Simply prints the error\msg in red
    :param err: The text that will be printed
    :return: None
    """
    print(colorama.Fore.RED + err + colorama.Fore.GREEN)


def find_free_slave() -> list:
    """
    Searches for an available slave (one that isn't working rn)
    :return: An available slave - [slave_socket, is_working_flag]
    """
    while True:  # Runs until there's a free slave
        if SLAVES:
            for slaves_manager in SLAVES:
                for slave in slaves_manager[2]:
                    if slave[1] == False:  # slave[1] is the 'working' flag -> if false, the slave is free
                        return slave


def kill_slave(slave: list) -> None:
    """
    Fixes an error with a slave\slave_manager: if the slave_manager is working, removing the problematic slave,
    else removes the slave_manager itself.
    :param slave: An [socket, bool] list that's represents a slave
    :return: None
    """
    for slaves_manager in SLAVES:
        if slave in slaves_manager[2]:
            try:
                slave_manager[0].send(b'')  # If the slave_manager isn't working, this will create an error
                if slave in slaves_manager[2]: slaves_manager[2].remove(slave)
            except:
                try:
                    SLAVES.remove(slaves_manager)
                    raise_exception('Fuck a slave manager got away')
                except:
                    pass
            break


def number_of_slavs() -> int:
    """
    Calculates the number of slaves that are working
    :return: The number of slaves
    """
    nos = 0
    for slave_manager in SLAVES:
        nos += len(slave_manager[2])
    return nos


def batch_mistake_value(net: NeuralNetwork, batch: Batch) -> float:
    mistake_sum = 0
    inputs = batch.images
    outputs = batch.labels

    for i in range(len(inputs)):
        mistake_sum += abs(net.run(inputs[i]) - outputs[i])

    return mistake_sum / batch.size


# ============================================== * MAIN * ==============================================

# ~~~~~~~~~~~~~~~ *** SLAVES MANAGEMENT *** ~~~~~~~~~~~~~~~
def chain_slaves(slaves_manager: list) -> None:
    """
    Adds the 'slave manager' slaves to the slave manager's list in SLAVES
    :param slaves_manager: A component in SLAVES that contains its slaves list
    :return: None
    """
    number_of_slaves = int.from_bytes(slaves_manager[0].recv(2), 'big')  # Receives the number of slaves in the slave
    # manager from it
    for _ in range(number_of_slaves):
        slaves_manager[2].append([SLAVES_COMMU.accept()[0], False])  # Adds a slave to the list


def chain_slave_managers() -> None:
    """
    Adds new 'slave managers' (computers) to the network - the SLAVES list.
    Every slave manager has slaves (threads) that are used in the training process, so it's basically:
    SLAVES[slave_manager[slave_1, slave_2...]...]
    This function is running all the time
    :return: None
    """
    while True:
        slaves_manager = [MAIN_COMMU.accept()[0], UPDATER_COMMU.accept()[0], []]
        # SLAVES[slave_manager[slave_manager_socket-1, slave_manager_socket-2, [slave[slave_socket, is_working]...]]...]
        try:
            send_net(slaves_manager[0])
            send_batch(slaves_manager, INPUTS, OUTPUTS)
            slaves_manager[0].send(pickle.dumps(net._bias_momentum) + pickle.dumps(net._weight_momentum))

            # number_of_slaves = slaves_manager[0].send(int.to_bytes(30, 2, 'big'))
            Thread(target=chain_slaves, args=(slaves_manager,), name='Slaves chainer').start()
            SLAVES.append(slaves_manager)
        except:
            if slaves_manager in SLAVES:
                SLAVES.remove(slaves_manager)


# ~~~~~~~~~~~~~~~ * SEND SHIT * ~~~~~~~~~~~~~~~

def send_net(slave_manager_socket: socket) -> None:
    """
    Sends the neural network to a slave manager
    :return: None
    """
    for layer in net.layers:
        bytes_layer = bytes(layer)
        slave_manager_socket.send(b'nxt')  # There is another layer
        activation_func = bytes(layer.activation_function_name, 'UTF8')
        slave_manager_socket.send(int.to_bytes(len(activation_func), 1, 'big') + activation_func)
        slave_manager_socket.send(int.to_bytes(len(bytes_layer), 5, 'big') + bytes_layer)
    slave_manager_socket.send(b'end')  # There are no other layers


def send_parameter_idx(slave_socket: socket, param_idx: tuple) -> None:
    """
    Sends a parameter's location to a slave. Parameter's indexes: [TYPE, layerIdx, paramIdx/sLayerIdx...]
    :param slave_socket: The slave's socket
    :param param_idx: The parameter's location
    :return: None.
    """
    param_idx: bytes = pickle.dumps(param_idx)
    slave_socket.send(int.to_bytes(len(param_idx), 1, 'big') + param_idx)


def send_batch(slaves_manager: list, inputs: list, outputs: list) -> None:
    """
    Sends a slaves_manager the inputs and true outputs.
    :param slaves_manager: The slave manager to which the data will be sent.
    :param inputs: The input.
    :param outputs: The true output.
    :return: None.
    """

    manager_socket = slaves_manager[0]
    try:
        manager_socket.send(b'INP')
        inp = pickle.dumps(inputs)
        out = pickle.dumps(outputs)
        manager_socket.send(int.to_bytes(len(inp), 8, 'big') + inp)
        manager_socket.send(int.to_bytes(len(out), 8, 'big') + out)
    except:
        try:
            manager_socket.send(b'')
        except:
            if manager_socket in SLAVES: SLAVES.remove(manager_socket)


# ~~~~~~~~~~~~~~~ * TRAINING PROCESS * ~~~~~~~~~~~~~~~

def use_slave(slave: list, param_idx: tuple) -> None:
    """
    Use the slave for the training process
    :param slave: A slave ([slave_socket, is_working_flag])
    :param param_idx: The parameter on which the slave should work
    :return: None
    """
    global progress

    slave[1] = True  # The slave is now working
    slave_socket: socket = slave[0]
    try:
        send_parameter_idx(slave_socket, param_idx)
    except:  # If there was a problem with communicating with the slave
        kill_slave(slave)
        exit()

    get_result(slave)
    progress += 1


def task_manager() -> None:
    """
    Goes through the network's parameters and working the slaves
    :return: None
    """
    global current_batch, current_param, INPUTS, OUTPUTS

    net.__mistakes_log__ = [batch_mistake_value(net=net, batch=test_batch)]

    while True:
        for slave_manager in SLAVES:
            send_batch(slave_manager, INPUTS, OUTPUTS)  # Sends the current inputs and outputs to all slave managers

        starting_time = time.time()

        for layer in reversed(net.layers):
            print(f'\nLayer: idx={layer.index}, size={layer.size}, time={time.time() - starting_time}')

            for bias_idx in range(layer.size):
                # All biases
                if update_param_flag: current_param = ('B', layer.index, bias_idx)
                Thread(target=use_slave, args=(find_free_slave(), ('B', layer.index, bias_idx))).start()

            for first_layer_idx in range(layer.size):
                # All the weights in the first layer
                for second_layer_idx in range(layer.prev_size):
                    # All the weights between the first and the second layer
                    if update_param_flag: current_param = ('W', layer.index, first_layer_idx, second_layer_idx)
                    Thread(target=use_slave,
                           args=(find_free_slave(), ('W', layer.index, first_layer_idx, second_layer_idx))).start()

        net.lets_be_loggin_shit(time=(time.time() - starting_time), number_of_slaves=number_of_slavs())
        current_batch = net.train_dataset[random.randint(0, len(net.train_dataset) - 1)]
        INPUTS = current_batch.images
        OUTPUTS = current_batch.labels


# SEND NETWORK'S PARAMETERS


def get_result(slave: list) -> None:
    """
    Get the feedback back from the slave (if the parameter should be changed or not), then activates 'fix_net'.
    :return: None.
    """
    slave_socket: socket = slave[0]
    try:
        new_val_b: bytes = slave_socket.recv(int.from_bytes(slave_socket.recv(3), 'big'))
        param_idx_b: bytes = slave_socket.recv(int.from_bytes(slave_socket.recv(4), 'big'))
    except:
        try:
            slave_socket.send(b'')
        except:
            kill_slave(slave)
            exit()
    fix_net(pickle.loads(new_val_b), pickle.loads(param_idx_b))
    Thread(target=update_slaves_nets, args=(param_idx_b, new_val_b)).start()

    slave[1] = False


def update_slaves_nets(param_idx_b: bytes, new_val_b: bytes):
    slave_manager = None
    try:

        for slave_manager in SLAVES:
            slave_manager[1].send(int.to_bytes(len(new_val_b), 3, 'big') + new_val_b + int.to_bytes(len(param_idx_b), 1,
                                                                                                    'big') + param_idx_b)
    except:
        if slave_manager in SLAVES:
            SLAVES.remove(slave_manager)
        raise_exception('F')


def fix_net(new_val: int, parameter_idx: tuple) -> None:
    """
    Change the parameter if it improves performance, and give the slave new task.
    :param should_change: True if changing the parameter is positive, False otherwise.
    :param parameter_idx: The parameter's indexes.
    :return: None
    """
    layer_idx = parameter_idx[1]
    if parameter_idx[0] == 'B':
        net.layers[layer_idx].biases[parameter_idx[2]] = new_val
    else:  # For Weights
        net.layers[layer_idx].weights[parameter_idx[2]][parameter_idx[3]] = new_val


# MAIN
if __name__ == '__main__':
    print('Starting...')
    Thread(target=chain_slave_managers, name='chainer', daemon=False).start()
    Thread(target=task_manager, name='task manager', daemon=False).start()
    last_time = time.time()
    last_progress = progress
    while True:
        cmd = input('Enter CMD: ')
        try:
            print(colorama.Fore.RED, end='')
            if 'add' in cmd:        # Add a slave in every slaves manager
                amount = int(cmd.strip('add '))
                for _ in range(amount):
                    for slave_manager in SLAVES:
                        slave_manager[0].send(b'ADD')
                        chain_slaves(slave_manager)

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
                inp = net.train_dataset[Bidx].images[idx]
                out = net.train_dataset[Bidx].labels[idx]
                net_output = net.run(inp)
                print(f'Location: ({Bidx},{idx})\n'
                      f'Inp: {str(inp)}\n'
                      f'Out: {out}\n'
                      f'Net: {net_output}\n'
                      f'MS: {utils.mistake_value(out, net_output)}')


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

            elif cmd == 'print progress':
                update_param_flag = True
                time.sleep(0.1)
                t = time.time()
                p = progress
                print(f'Param: {current_param}, Idx: {p}\nTime: {t - last_time} [S]\n'
                      f'Progress: {(p - last_progress) / (t - last_time)} [P/S]')
                update_param_flag = False
                last_time = t
                last_progress = p

            elif cmd == 'print mistake':
                print(batch_mistake_value(net, test_batch))

            elif cmd == 'graph':
                try:
                    net.graph(temp=True)
                except Exception as e:
                    print(e)
            elif cmd == 'exit':
                break

            else:
                print(f'\'{cmd}\' ~ illegal command')

        except:
            print(f'\'{cmd}\' ~ illegal command')
        print(colorama.Fore.GREEN, end='')
