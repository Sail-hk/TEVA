import torch
import random
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from models.Nets import CNNMnist, MLP, CNNCifar
import copy
import time
from charm.toolbox.pairinggroup import PairingGroup, ZR, G1, pair, GT
from threshold_paillier import *
from pympler import asizeof

group = PairingGroup('SS512')

def set_requires_grad(net: nn.Module, mode=True):
    for p in net.parameters():
        p.requires_grad_(mode)

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class Client():
    
    def __init__(self, args, dataset=None, idxs=None, w=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.img_size = dataset[0][0].shape
        self.predictions = []
        if args.model_name == 'cnn' and args.dataset == 'cifar':
            self.model = CNNCifar(args=args).to(args.device)
        elif args.model_name == 'cnn' and args.dataset == 'mnist':
            self.model = CNNMnist(args=args).to(args.device)
        elif args.model_name == 'mlp':
            len_in = 1
            for x in self.img_size:
                len_in *= x
            self.model = MLP(dim_in=len_in, dim_hidden=args.dim_hidden, dim_out=args.num_classes).to(args.device)
        else:
            exit('Error: unrecognized model')
        self.model.load_state_dict(w)
        self.m_c = {k: torch.zeros_like(v) for k, v in self.model.state_dict().items()}
        # DP hyperparameters
        self.C = self.args.C
    def train(self, h, sk, g, H_0, t, pub_key):
        w_old = copy.deepcopy(self.model.state_dict())
        V_t = copy.deepcopy(w_old)
        A_t = copy.deepcopy(w_old)
        C_t = copy.deepcopy(w_old)
        net = copy.deepcopy(self.model)
        net.train()   
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        total = 0
        correct = 0
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
                _, predicted = torch.max(log_probs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total

        f = random.random()
        w_new = net.state_dict()

        update_w = {}
        if self.args.mode == 'plain':
            for k in w_new.keys():
                update_w[k] = w_new[k] - w_old[k]
                
        elif self.args.mode == 'DP':
            for k in w_new.keys():
                update_w[k] = w_new[k] - w_old[k]
                # L2-norm
                sensitivity = torch.norm(update_w[k], p=2)
                # clip
                update_w[k] = update_w[k] / max(1, sensitivity / self.C)

        elif self.args.mode == 'Threshold Paillier':
            if t % 3 == 0:
                print("Mode I start............")
                #start = time.time()
                for k in w_new.keys():
                    update_w[k] = w_new[k] - w_old[k]
                    for i in range(len(self.m_c[k])):
                        self.m_c[k][i] = self.args.client_grammar * self.m_c[k][i] + self.args.lr * update_w[k][i]
                    update_w[k] = update_w[k] - self.m_c[k]

                    list_w = update_w[k].view(-1).cpu().tolist()

                    v = V_t[k].view(-1).cpu().tolist()
                    a_t = A_t[k].view(-1).cpu().tolist()
                    c = C_t[k].view(-1).cpu().tolist()
                    for i, elem in enumerate(list_w):
                        v[i] = pow(H_0, sk) # 这里的表示有问题！ v[i]应该是 pow(H_0, sk) * pow(h, group.random(ZR, elem if elem>=1 else 1) * weig_i_t)
                        # testone = type(group.random(ZR, elem) if elem>=1 else 1)
                        # testone2 = type(sk)
                        # testone3 = group.random(ZR, elem)
                        # print("testone", testone)
                        # print("testone2", testone2)
                        # print("h", type(h))
                        # print("H_0", type(H_0))
                        c[i] = pow(h,group.random(ZR,elem if elem>=1 else 1))
                        list_w[i] = pub_key.encrypt(elem) # 小密钥解密出问题和这里可能有很大关系
                        a_t[i] = pow(g, sk)
                    update_w[k] = list_w
                    V_t[k] = v
                    A_t[k] = a_t
                    C_t[k] = c
                #print("v_t", V_t)
            else:
                print("Mode II start............")
                for k in w_new.keys():
                    update_w[k] = w_new[k] - w_old[k]
                    # flatten weight
                    list_w = update_w[k].view(-1).cpu().tolist()
                    v = V_t[k].view(-1).cpu().tolist()
                    a_t = A_t[k].view(-1).cpu().tolist()
                    for i, elem in enumerate(list_w):
                        v[i] = pow(H_0, sk) * pow(h, group.random(ZR, elem if elem>=1 else 1))
                        list_w[i] = pub_key.encrypt(elem)
                        a_t[i] = pow(g, sk)

                    update_w[k] = list_w
                    V_t[k] = v
                    A_t[k] = a_t
        else:
            raise NotImplementedError

        return update_w, sum(batch_loss) / len(batch_loss), V_t, A_t, accuracy, f, C_t
    def verify(self, verify_w_glob, V_glob, g, H_0_t, A_glob, h, t):
        verify_w = copy.deepcopy(verify_w_glob)
        num_verif = 0
        for k in verify_w_glob.keys():
            list_verify = verify_w[k].view(-1).cpu().tolist()
            for i, elem in enumerate(list_verify):
                list_verify[i] = group.init(ZR, elem)
            verify_w[k] = list_verify

        num_pair = 0
        if t % 3 == 0: # 因为前面训练的时候A的计算和论文里不一样，所以这里验证一定是错误的
            for k in V_glob.keys():
                num_pair += len(V_glob[k])
                for j in range(len(V_glob[k])):
                    if pair(V_glob[k][j], g) == pair(pow(H_0_t, group.init(ZR, -self.args.lr)), A_glob[k][j]) * pair(h, pow(g, verify_w[k][j])):
                        num_verif += 1
                    else:
                        print("Pairing validation error!")
                        # exit()
        else:
            for k in V_glob.keys():
                num_pair += len(V_glob[k])
                for j in range(len(V_glob[k])):
                    if pair(V_glob[k][j], g) == pair(pow(H_0_t, group.init(ZR, 1/self.args.num_users)), A_glob[k][j]) * pair(h, pow(g, verify_w[k][j])):
                        num_verif += 1
                    else:
                        print("Pairing validation error!")
                        # exit()

        if num_verif == num_pair:
            return True
        else:
            print("validation failed!".format(iter))
            # exit()

    def update(self, w_glob, publ_key, shares):
        update_w_avg = copy.deepcopy(w_glob)
        seleted_shares = random.sample(shares, self.args.threshold)
        recon_priv_key = PaillierPrivateKey.reconstruct_key(seleted_shares, publ_key)
        if self.args.mode == 'plain' or self.args.mode == 'DP':
            self.model.load_state_dict(w_glob)
        elif self.args.mode == 'Threshold Paillier':

            print('decrypting...')
            for k in update_w_avg.keys():

                for i, elem in enumerate(update_w_avg[k]):
                    update_w_avg[k][i] = recon_priv_key.decrypt(elem)
                origin_shape = list(self.model.state_dict()[k].size())
                update_w_avg[k] = torch.FloatTensor(update_w_avg[k]).to(self.args.device).view(*origin_shape)
                self.model.state_dict()[k] += update_w_avg[k]

        else:
            raise NotImplementedError
