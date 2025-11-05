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
import torch.nn.functional as F

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
        
   # ===== 这是新的 train 函数 (快速版) =====
    def train(self, h, sk, g, H_0, t, pub_key, global_delta_w):
        w_old = copy.deepcopy(self.model.state_dict())
        V_t = copy.deepcopy(w_old)
        A_t = copy.deepcopy(w_old)
        C_t = copy.deepcopy(w_old) # 尽管我们不用 C_t, 但保留它以匹配返回值
        net = copy.deepcopy(self.model)
        net.train()   
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        total = 0
        correct = 0
        batch_loss = []
        for iter in range(self.args.local_ep):
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

        # 计算 sim (保持不变)
        local_delta_w_vec = []
        global_delta_w_vec = []
        for k in w_new.keys():
            delta_w_k = w_new[k] - w_old[k]
            local_delta_w_vec.append(delta_w_k.view(-1))
            global_delta_w_vec.append(global_delta_w[k].view(-1))
        local_delta_w_vec = torch.cat(local_delta_w_vec)
        global_delta_w_vec = torch.cat(global_delta_w_vec)
        sim = 0.0
        if t > 0:
            sim = F.cosine_similarity(local_delta_w_vec, global_delta_w_vec, dim=0, eps=1e-8).item()

        if self.args.mode == 'plain':
            for k in w_new.keys():
                update_w[k] = w_new[k] - w_old[k]

        elif self.args.mode == 'DP':
            for k in w_new.keys():
                update_w[k] = w_new[k] - w_old[k]
                sensitivity = torch.norm(update_w[k], p=2)
                update_w[k] = update_w[k] / max(1, sensitivity / self.C)

        elif self.args.mode == 'Threshold Paillier':
            if t % 3 == 0:
                print("Mode I start............ (Crypto Disabled)") # 修改提示
                for k in w_new.keys():
                    # OdMum 动量 (保持不变)
                    update_w[k] = w_new[k] - w_old[k]
                    # 添加这两行 (正确的张量计算)
                    delta_w_k = update_w[k] # update_w[k] 此时等于 w_new[k] - w_old[k]
                    self.m_c[k] = self.args.client_grammar * self.m_c[k] + self.args.lr * delta_w_k
                    update_w[k] = delta_w_k - self.m_c[k]

                    list_w = update_w[k].view(-1).cpu().tolist()
                    v = V_t[k].view(-1).cpu().tolist()
                    a_t = A_t[k].view(-1).cpu().tolist()
                    c = C_t[k].view(-1).cpu().tolist()

                    for i, elem in enumerate(list_w):
                        # ===== 密码学操作 (已禁用) =====
                        # v[i] = pow(H_0, sk) 
                        # c[i] = pow(h,group.random(ZR,elem if elem>=1 else 1))
                        list_w[i] = elem # <--- 关键修改：不加密
                        # a_t[i] = pow(g, sk)
                        # ===== 结束 =====

                    update_w[k] = list_w
                    V_t[k] = v # v 和 a_t 现在只是空的
                    A_t[k] = a_t
                    C_t[k] = c
            else:
                print("Mode II start............ (Crypto Disabled)") # 修改提示
                for k in w_new.keys():
                    update_w[k] = w_new[k] - w_old[k]
                    list_w = update_w[k].view(-1).cpu().tolist()
                    v = V_t[k].view(-1).cpu().tolist()
                    a_t = A_t[k].view(-1).cpu().tolist()

                    for i, elem in enumerate(list_w):
                        # ===== 密码学操作 (已禁用) =====
                        # v[i] = pow(H_0, sk) * pow(h, group.random(ZR, elem if elem>=1 else 1))
                        list_w[i] = elem # <--- 关键修改：不加密
                        # a_t[i] = pow(g, sk)
                        # ===== 结束 =====

                    update_w[k] = list_w
                    V_t[k] = v # v 和 a_t 现在只是空的
                    A_t[k] = a_t
        else:
            raise NotImplementedError

        return update_w, sum(batch_loss) / len(batch_loss), V_t, A_t, accuracy, f, C_t, sim

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

        # ===== 这是新的 update 函数 (快速版) =====
    # ===== 这是最终版的 update 函数 (V3 - 快速版) =====
    def update(self, w_glob, publ_key, shares):
        # w_glob 现在是【完整的模型状态】，不再是增量

        if self.args.mode == 'plain' or self.args.mode == 'DP':
            self.model.load_state_dict(w_glob)
        elif self.args.mode == 'Threshold Paillier':

            # 我们不再需要解密，因为服务器发送的是明文模型
            print('Loading new global model (plaintext)...') # 修改提示

            # ===== 关键修改：直接加载服务器发来的新模型 =====
            self.model.load_state_dict(w_glob)
            # ===== 结束 =====

        else:
            raise NotImplementedError