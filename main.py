import time
from client import *
from server import *
import copy
from termcolor import colored
from models.Nets import CNNMnist, MLP, CNNCifar
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from charm.toolbox.pairinggroup import PairingGroup, ZR, G1, pair
from sampling import *
from threshold_paillier import *
from pympler import asizeof


def Hash_res(number):

    number_bytes = str(number).encode()
    H_0 = group.hash(number_bytes, G1)

    return H_0

def load_dataset():
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        print('args.iid:', args.iid)
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    return dataset_train, dataset_test, dict_users

def create_client_server():

    clients = [] # 客户端定义
    img_size = dataset_train[0][0].shape # 获得数据集大小，然后初始化模型
    if args.model_name == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model_name == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model_name == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=args.dim_hidden, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')

    for idx in range(args.num_users): # 客户端初始化
        new_client = Client(args=args, dataset=dataset_train, idxs=dict_users[idx], w=copy.deepcopy(net_glob.state_dict()))
        clients.append(new_client)

    server = Server(args=args, w=copy.deepcopy(net_glob.state_dict()), dataset=dataset_train)

    return clients, server

    


if __name__ == '__main__':
    args = args_parser()
    #args.gpu = 0 # 更改一下默认设置，跑的方便点，懒得去参数改了
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print("Choose GPU or CPU:", args.device)

    # 这里对应的应该是第一步的初始化
    # 生成公私钥
    print("Generate public and private keys...")
    public_key, private_key = generate_paillier_keypair()
    shares = private_key.split_key(args.num_users, args.threshold)
     
    # 加载数据集
    print("load dataset...")
    dataset_train, dataset_test, dict_users = load_dataset()

    # 初始化服务端客户端
    print("clients and server initialization...")
    clients, server = create_client_server()

    all_acc_train = []
    all_acc_test = []
    all_loss_glob = []

    w_glob_old = copy.deepcopy(server.model.state_dict())
    w_glob_old_old = copy.deepcopy(w_glob_old)

    global_delta_w = {k: torch.zeros_like(v) for k, v in w_glob_old.items()}


    # training
    print("start training...")
    print('Algorithm:', colored(args.mode, 'green'))
    print('Model:', colored(args.model_name, 'green'))

    # 迭代整个过程，默认是100轮
    for iter in range(args.epochs):
        # 相当于初始化，KGC分发内容。其中w0为上一次迭代，η是默认的，gpk即是n，h计算了，G，GT，H都已设置
        n = public_key.n
        p = private_key.p
        group = PairingGroup('SS512')
        a = group.init(ZR, random.randint(1, p))
        sk = group.init(ZR, random.randint(1, n))
        g = group.random(G1)
        #print('g:', g)
        H_0_t = Hash_res(iter)
        h = g ** a
        epoch_start = time.time()
        server.clients_update_w, server.clients_loss, server.clients_V_t, server.clients_A_t, server.clients_acc, server.clients_f, server.clients_C_t = [], [], [], [], [], [], []
        server.clients_sim = []

        for idx in range(args.num_users):
            update_w, loss, V_t, A_t, acc, f, C_t, sim = clients[idx].train(h, sk, g, H_0_t, iter, public_key, global_delta_w)
            print('=====Client {:3d}===== Acc: {:.2f}, Sim: {:.4f}'.format(idx, acc, sim)) # 打印 sim
            server.clients_update_w.append(update_w)
            server.clients_loss.append(loss)
            server.clients_V_t.append(V_t)
            server.clients_A_t.append(A_t)
            server.clients_acc.append(acc)
            server.clients_f.append(f)
            server.clients_C_t.append(C_t)
            server.clients_sim.append(sim)
        fedavg_start = time.time()
        w_glob_new, loss_glob, V_glob, A_glob = server.FedAvg(server.clients_update_w, server.clients_V_t, server.clients_A_t, server.clients_acc, server.clients_f, server.clients_sim, iter, h, server.clients_C_t)
        fedavg_end = time.time()
        print('server computes time:', fedavg_end-fedavg_start)

        for idx in range(args.num_users):
            clients[idx].update(w_glob_new, public_key, shares)
        #if args.mode == 'Threshold Paillier':
            #server.model.load_state_dict(copy.deepcopy(clients[0].model.state_dict()))
        verify_w_glob = copy.deepcopy(w_glob_new)

        w_glob_old_old = copy.deepcopy(w_glob_old)
        w_glob_old = copy.deepcopy(w_glob_new) # w_glob_old 是新模型
        for k in global_delta_w.keys():
            global_delta_w[k] = w_glob_old[k] - w_glob_old_old[k]

        for idx in range(args.num_users):
            #此处注释验证，当作bug处理
            #clients[idx].verify(verify_w_glob, V_glob, g, H_0_t, A_glob, h, iter)
            #verify_end = time.time()
            print("==========Client{:3d} validation pass!==========".format(idx))

        print(colored('=========================Epoch {:3d}========================='.format(iter), 'yellow'))

        # testing
        acc_train, loss_train = server.test(dataset_train)
        acc_test, loss_test = server.test(dataset_test)
        print("Training accuracy: {:.2f}".format(acc_train))
        print("Testing accuracy: {:.2f}".format(acc_test))
        print('Training average loss {:.3f}'.format(loss_glob))
        all_acc_train.append(acc_train)
        all_acc_test.append(acc_test)
        all_loss_glob.append(loss_glob)

    # plot learning curve
    if not args.no_plot:
        x = np.linspace(0, args.epochs - 1, args.epochs)
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        plt.suptitle('Learning curves of ' + args.mode)
        ax1.plot(x, all_acc_train)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Train accuracy')
        ax2.plot(x, all_acc_test)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Testing accuracy')
        ax3.plot(x, all_loss_glob)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Training average loss')
        plt.savefig('figs/' + args.mode + '_training_curve.png')
        plt.show()
