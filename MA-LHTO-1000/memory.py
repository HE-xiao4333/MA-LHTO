import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from  torch.distributions import Categorical
from config import config
# 初始化函数
# 用户 Agent 网络
'''
class UserAgentNetwork(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(UserAgentNetwork,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0., 0.3)
                nn.init.constant_(layer.bias, 0.1)
    def forward(self,h):
        m_pred = self.model(h)  # 输入h得到预测结果m_pred
        #m_pred = m_pred.detach().numpy()
        return m_pred'''

# 云 Agent 网络
class UserAgentNetwork(nn.Module):
    def __init__(self, input_dim,output):
        super(UserAgentNetwork,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output),
            nn.Sigmoid()
        )
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

    def forward(self, x):
        if torch.cuda.is_available():
            x = x.to("cuda")
        x = self.model(x)
        return x

class multiagent(object):
    def __init__(self,Dev_number,state_dim_Dev,action_dim_xy,action_dim_z,action_dim_f,learning_rate=0.001, training_interval=0.5, batch_size=64,memory_size=100000,entropy_weight=0.01):
        super(multiagent, self).__init__()
        self.Dev = Dev_number
        self.state_dim_ve = state_dim_Dev
        self.action_dim_xy = action_dim_xy
        self.action_dim_z = action_dim_z
        self.action_dim_f = action_dim_f
        self.memory_size = memory_size
        self.training_interval = training_interval  # learn every #training_interval
        self.lr = learning_rate
        self.batch_size = batch_size
        self.entropy_weight=entropy_weight
        # reply buffer
        self.memory_Dev1 = np.zeros((self.memory_size, state_dim_Dev + 3))  # 输入、输出、第几个dev、reward、总reward
        self.point_Dev1 = 0
        self.memory_Dev2 = np.zeros((self.memory_size, state_dim_Dev + 3))  # 输入、输出、第几个dev、reward、总reward
        self.point_Dev2 = 0
        self.memory_Dev3 = np.zeros((self.memory_size, state_dim_Dev + action_dim_f))  # 输入、输出、第几个dev、reward、总reward
        self.point_Dev3 = 0

        self.user_agent1=[]
        self.user_agent1_target = []
        self.copt_user1 = []
        self.cost_Dev=[]
        for _ in range(self.Dev):
            user_agent = UserAgentNetwork(self.state_dim_ve, self.action_dim_xy)
            user_target = UserAgentNetwork(self.state_dim_ve, self.action_dim_xy)
            copt = torch.optim.Adam(user_agent.parameters(), lr=0.001, betas=(0.09, 0.999), weight_decay=0.0001)
            self.user_agent1.append(user_agent)
            self.user_agent1_target.append(user_target)
            self.copt_user1.append(copt)
            self.cost_Dev.append([])
        self.user_agent2 = []
        self.user_agent2_target = []
        self.copt_user2 = []
        for _ in range(self.Dev):
            user_agent = UserAgentNetwork(self.state_dim_ve, self.action_dim_z)
            user_target = UserAgentNetwork(self.state_dim_ve, self.action_dim_z)
            copt = torch.optim.Adam(user_agent.parameters(), lr=0.001, betas=(0.09, 0.999), weight_decay=0.0001)
            self.user_agent2.append(user_agent)
            self.user_agent2_target.append(user_target)
            self.copt_user2.append(copt)
        self.user_agent3 = []
        self.user_agent3_target = []
        self.copt_user3 = []
        for _ in range(self.Dev):
            user_agent = UserAgentNetwork(self.state_dim_ve, self.action_dim_f)
            user_target = UserAgentNetwork(self.state_dim_ve, self.action_dim_f)
            copt = torch.optim.Adam(user_agent.parameters(), lr=0.001, betas=(0.09, 0.999), weight_decay=0.0001)
            self.user_agent3.append(user_agent)
            self.user_agent3_target.append(user_target)
            self.copt_user3.append(copt)

    def choose_action_xy(self, j,s):
        s = torch.FloatTensor(s)
        self.user_agent1[j].eval()
        action = self.user_agent1[j](s)
        if torch.cuda.is_available():
            action = action.cpu()
        action = F.softmax(action, dim=0)
        action = action.detach().numpy()
        action_all=action.tolist()
        return action_all
    def choose_action_z(self, j,s):
        s = torch.FloatTensor(s)
        self.user_agent2[j].eval()
        action = self.user_agent2[j](s)
        if torch.cuda.is_available():
            action = action.cpu()
        action = F.softmax(action, dim=0)
        action = action.detach().numpy()
        action_all=action.tolist()
        return action_all
    def choose_action_f(self, j,s,time):
        s = torch.FloatTensor(s)
        self.user_agent3[j].eval()
        action = self.user_agent3[j](s)
        if torch.cuda.is_available():
            action = action.cpu()
        action = action.detach().numpy()
        action_all=action.tolist()
        if time <50:
            k=1
        elif time<100:
            k=1
        else:
            k=1
        #action_target = self.reverse(action_all,time,k)
        return action_all

    def remember_Dev1(self,ID,state,action,r):
        idx = self.point_Dev1 % self.memory_size
        if r > 0:
            self.memory_Dev1[int(idx), :] = np.hstack((ID, state, action,r))
            self.point_Dev1 += 1
            if (self.point_Dev1 % (self.training_interval*config.get("Dev_dev")) == 0):
                self.learn_Dev1()

    def remember_Dev2(self,ID,state,action,r):
        idx = self.point_Dev2 % self.memory_size
        if r > 0:
            self.memory_Dev2[int(idx), :] = np.hstack((ID, state, action,r))
            self.point_Dev2 += 1
            if (self.point_Dev2 % (self.training_interval*config.get("Dev_dev")) == 0):
                self.learn_Dev2()

    def remember_Dev3(self,ID,state,action,r):
        idx = self.point_Dev3 % self.memory_size
        if r > 0:
            self.memory_Dev3[int(idx), :] = np.hstack((state, action))
            self.point_Dev3 += 1
            if (self.point_Dev3 % (self.training_interval*config.get("Dev_dev")) == 0):
                self.learn_Dev3()

    def learn_Dev1(self):
        #criterion = nn.BCELoss()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for i in range(config.get('Dev_dev')):
            if self.point_Dev1 > self.memory_size:
                sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            else:
                sample_index = np.random.choice(self.point_Dev1, size=self.batch_size)
            batch_memory = self.memory_Dev1[sample_index, :]
            h_train = torch.tensor(batch_memory[:, 1: self.state_dim_ve + 1], dtype=torch.float32).to(device)
            m_train = torch.tensor(batch_memory[:, self.state_dim_ve + 1:-1], dtype=torch.int64).to(device)
            reward = torch.tensor(batch_memory[:, -1], dtype=torch.int64).to(device)
            self.user_agent1[i].train()
            self.copt_user1[i].zero_grad()
            action_probs = self.user_agent1[i](h_train)
            dist = Categorical(logits=action_probs)
            log_probs = dist.log_prob(m_train.squeeze(1))  # calculate log probabilities for chosen actions
            # Optionally, compute entropy for monitoring or additional purposes
            entropy = dist.entropy().mean()  # mean entropy across the batch
            loss = -torch.mean(log_probs*reward) - 0.1* torch.mean(entropy)  # typical RL loss using log probabilities
            #loss = criterion(predict, m_train)
            loss.backward()
            self.copt_user1[i].step()
            self.cost = loss.item()
            assert (self.cost > 0)
            self.cost_Dev[i].append(self.cost)  #
            tau = 0.99
            if (self.point_Dev1 % (5*self.training_interval*config.get("Dev_dev")) == 0):
                nn.init.normal_(self.user_agent1_target[i].model[-4].weight, mean=0., std=0.3)
                nn.init.constant_(self.user_agent1_target[i].model[-4].bias, 0.1)
            for param, target_param in zip(self.user_agent1[i].parameters(), self.user_agent1_target[i].parameters()):
                    param.data.copy_((1 - tau) * param.data + tau * target_param.data)
            self.user_agent1[i].eval()

    def learn_Dev2(self):
        #criterion = nn.BCELoss()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for i in range(config.get('Dev_dev')):
            if self.point_Dev2 > self.memory_size:
                sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            else:
                sample_index = np.random.choice(self.point_Dev2, size=self.batch_size)
            batch_memory = self.memory_Dev2[sample_index, :]
            h_train = torch.tensor(batch_memory[:, 1: self.state_dim_ve + 1], dtype=torch.float32).to(device)
            m_train = torch.tensor(batch_memory[:, self.state_dim_ve + 1:-1], dtype=torch.int64).to(device)
            reward = torch.tensor(batch_memory[:, -1], dtype=torch.int64).to(device)
            self.user_agent2[i].train()
            self.copt_user2[i].zero_grad()
            action_probs = self.user_agent2[i](h_train)
            dist = Categorical(logits=action_probs)
            log_probs = dist.log_prob(m_train.squeeze(1))  # calculate log probabilities for chosen actions
            # Optionally, compute entropy for monitoring or additional purposes
            entropy = dist.entropy().mean()  # mean entropy across the batch
            loss = -torch.mean(log_probs*reward) - 0.1* torch.mean(entropy)  # typical RL loss using log probabilities
            #loss = criterion(predict, m_train)
            loss.backward()
            self.copt_user2[i].step()
            self.cost = loss.item()
            assert (self.cost > 0)
            #self.cost_Dev[i].append(self.cost)  #
            tau = 0.99
            if (self.point_Dev2 % (5*self.training_interval*config.get("Dev_dev")) == 0):
                nn.init.normal_(self.user_agent2_target[i].model[-4].weight, mean=0., std=0.3)
                nn.init.constant_(self.user_agent2_target[i].model[-4].bias, 0.1)
            for param, target_param in zip(self.user_agent2[i].parameters(), self.user_agent2_target[i].parameters()):
                    param.data.copy_((1 - tau) * param.data + tau * target_param.data)
            self.user_agent2[i].eval()

    def learn_Dev3(self):
        criterion = nn.BCELoss()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for i in range(config.get('Dev_dev')):
            if self.point_Dev3 > self.memory_size:
                sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            else:
                sample_index = np.random.choice(self.point_Dev3, size=self.batch_size)
            batch_memory = self.memory_Dev3[sample_index, :]
            h_train = torch.tensor(batch_memory[:, 0: self.state_dim_ve], dtype=torch.float32).to(device)
            m_train = torch.tensor(batch_memory[:, self.state_dim_ve:], dtype=torch.float32).to(device)
            self.user_agent3[i].train()
            self.copt_user3[i].zero_grad()
            predict = self.user_agent3[i](h_train)
            loss = criterion(predict, m_train)
            loss.backward()
            self.copt_user3[i].step()
            self.cost = loss.item()
            assert (self.cost > 0)
            self.cost_Dev[i].append(self.cost)  #
            tau = 0.99
            if (self.point_Dev3 % (5*self.training_interval*config.get("Dev_dev")) == 0):
                nn.init.normal_(self.user_agent3_target[i].model[-4].weight, mean=0., std=0.3)
                nn.init.constant_(self.user_agent3_target[i].model[-4].bias, 0.1)
            for param, target_param in zip(self.user_agent3[i].parameters(), self.user_agent3_target[i].parameters()):
                    param.data.copy_((1 - tau) * param.data + tau * target_param.data)
            self.user_agent3[i].eval()

'''
    def reverse(self, m, time,k=3):
        m_list = []
        if k > 0:
            for i in range(k):
                c = []
                # 产生随机数
                if time <70:
                    nu = np.random.uniform(-0.5, 0.5)
                elif time <100:
                    nu = np.random.uniform(-0.25, 0.25)
                else:
                    nu = np.random.uniform(-0.15, 0.15)
                for j in range(len(m)):
                    if (m[j] + nu < 0.01):
                        index = 0.01
                    elif (m[j] + nu > 0.99):
                        index = 0.99
                    else:
                        index = m[j] + nu
                    c.append(index)
                m_list.append(c)
            pass
        return m_list
'''
class SimpleNeuralNetwork(nn.Module):
    def __init__(self,input,output):
        super(SimpleNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

class unit_all(object):
    def __init__(self,state_dim_input_mes,state_dim_output_mes):
        super(unit_all, self).__init__()
        self.model_mes = SimpleNeuralNetwork(state_dim_input_mes,state_dim_output_mes)
    def get_output_mes(self,input):
        input = torch.tensor(input, dtype=torch.float32)
        self.model_mes.eval()
        output = self.model_mes(input)
        output = output.detach().numpy()
        return output