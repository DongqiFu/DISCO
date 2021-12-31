import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score


if torch.cuda.is_available():  
    dev = "cuda:0" 
else:  
    dev = "cpu" 
device = torch.device(dev)


class CNN_1d(nn.Module):
    def __init__(self, hidden, kernel_s):
        super(CNN_1d, self).__init__()
        self.conv1 = nn.Conv1d(1, 20, kernel_size =kernel_s)
        self.conv2 = nn.Conv1d(20, 40, kernel_size =kernel_s)
        self.fc1 = nn.Linear(hidden, 128)
        self.fc2 = nn.Linear(128, 1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



class mufasa:
    def __init__(self,  hidden = 100, in_channel =1, lamdba=0.0001, nu=0.00001, if_2d = 1, kernel_s =2, lr = 0.001, trounds = 5000):
        self.func = CNN_1d(hidden, kernel_s).to(device)
        self.context_list = []
        self.reward = []
        self.lamdba = lamdba
        self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)
        self.U = lamdba * torch.ones((self.total_param,)).to(device)
        self.nu = nu
        self.loss = torch.nn.MSELoss()
        self.lr = lr
        self.trounds= trounds

    def select(self, context):
        context = torch.from_numpy(context).float()
        context = torch.unsqueeze(context, 1)
        context = context.to(device)
        context.requires_grad = True
        mu = self.func(context.to(device))
        g_list = []
        sampled = []
        ave_sigma = 0
        ave_rew = 0
        f_l = []
        s_l = []
        index = 0
        for fx in mu:
            self.func.zero_grad()
            fx.backward(retain_graph=True)
            g = torch.cat([p.grad.flatten().detach() for p in self.func.parameters()])
            g_list.append(g)
            sigma2 = self.lamdba * self.nu * g * g / self.U
            sigma = torch.sqrt(torch.sum(sigma2))
            sample_r = fx.item() + sigma.item()
            sampled.append(sample_r)
            ave_sigma += sigma.item()
            ave_rew += sample_r
            f_l.append(fx.item())
            s_l.append(sigma.item())
            index +=1
        in_g = context.grad
        arm = np.argmax(sampled)
        self.U += g_list[arm] * g_list[arm]
        return arm, sampled
    
    def predict_prob(self, context):
        context = torch.from_numpy(context).float()
        context = torch.unsqueeze(context, 1)
        context = context.to(device)
        context.requires_grad = True
        mu = self.func(context.to(device))
        g_list = []
        sampled = []
        ave_sigma = 0
        ave_rew = 0
        f_l = []
        s_l = []
        index = 0
        for fx in mu:
            self.func.zero_grad()
            fx.backward(retain_graph=True)
            g = torch.cat([p.grad.flatten().detach() for p in self.func.parameters()])
            g_list.append(g)
            sigma2 = self.lamdba * self.nu * g * g / self.U
            sigma = torch.sqrt(torch.sum(sigma2))
            sample_r = fx.item() + sigma.item()
            sampled.append(sample_r)
            ave_sigma += sigma.item()
            ave_rew += sample_r
            f_l.append(fx.item())
            s_l.append(sigma.item())
            index +=1
        in_g = context.grad
        self.U += g_list[0] * g_list[0]
        s = sampled[0]
        if s > 1.0:
            s = 1.0
        if s < 0.0:
            s = 0.0
        return s
    
  
    
    def update(self, context, arm_select, reward):
        context = torch.from_numpy(context).float()
        context = torch.unsqueeze(context, 1)
        con = torch.unsqueeze(context[arm_select], 0)
        self.context_list.append(con)
        self.reward.append(reward)


    def train(self):
        #optimizer = optim.SGD(self.func.parameters(), lr=self.lr)
        optimizer = optim.Adam(self.func.parameters(), lr=1e-4)
        length = len(self.reward)
        index = np.arange(length)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0
        while True:
            batch_loss = 0
            cnt_w = 0
            for idx in index:
                c = self.context_list[idx]
                r = self.reward[idx]
                r = torch.tensor([r]).float().to(device)
                optimizer.zero_grad()
                output = self.func(c.to(device))
                loss = self.loss(output, r)
                #loss = (output - r)**2
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= self.trounds:
                    return tot_loss / self.trounds
            if batch_loss / length <= 0.001:
                return batch_loss / length  
            

class mufasa_classifier:
    def __init__(self):
        self.model = mufasa(hidden = 11920)

    def train_mufasa(self, X_train, y_train):
        train_s = len(X_train)
        for i in range(train_s):
            self.model.update(np.array([X_train[i]]), 0, y_train[i])
            if i<100:
                if i%10 == 0:
                    loss = self.model.train()
            else:
                if i%100 == 0:
                    loss = self.model.train()
                    
    def test_mufasa(self, X_test, y_test, thre = 0.4):
        test_s = len(X_test)
        accu = []
        for i in range(test_s):
            context = np.array([X_test[i]])
            p = self.model.predict_prob(context)
            #print(p)
            if p > thre:
                accu.append(1)
            else:
                accu.append(0)
        return accuracy_score(accu, y_test, normalize=True), accu
    
    def predict_prob(self, X):
        context = np.array([X])
        return self.model.predict_prob(context)
    
    
class mufasa_classifier_20:
    def __init__(self):
        self.model = mufasa(hidden = 720)

    def train_mufasa(self, X_train, y_train):
        train_s = len(X_train)
        for i in range(train_s):
            self.model.update(np.array([X_train[i]]), 0, y_train[i])
            if i<100:
                if i%10 == 0:
                    loss = self.model.train()
            else:
                if i%100 == 0:
                    loss = self.model.train()
                    
    def test_mufasa(self, X_test, y_test, thre = 0.4):
        test_s = len(X_test)
        accu = []
        for i in range(test_s):
            context = np.array([X_test[i]])
            p = self.model.predict_prob(context)
            #print(p)
            if p > thre:
                accu.append(1)
            else:
                accu.append(0)
        return accuracy_score(accu, y_test, normalize=True), accu
    
    def predict_prob(self, X):
        context = np.array([X])
        return self.model.predict_prob(context)