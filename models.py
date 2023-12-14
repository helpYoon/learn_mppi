import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.func import jacrev
from collections import deque
import random

# Experience replay buffer
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device
    
    def push(self, o, a, r, o_1):
        self.buffer.append((o, a, r, o_1))
    
    def sample_transitions(self, batch_size):
        O, A, R, O_1 = zip(*random.sample(self.buffer, batch_size))
        return torch.stack(O).double(), \
                torch.stack(A).double(), \
                torch.stack(R).double(), \
                torch.stack(O_1).double()

    def sample_states(self, batch_size):
        O, A, R, O_1 = zip(*random.sample(self.buffer, batch_size))
        return torch.stack(O).double()

    def __len__(self):
        return len(self.buffer)
    
# DNN dynamics model
class dnn(torch.nn.Module):
    def __init__(self, obs_size, action_size):
        super(dnn, self).__init__()
        self.fc1 = torch.nn.Linear(obs_size+action_size, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, obs_size)

    def forward(self, x, a):
        concat = torch.cat((x,a),-1)
        y1 = F.relu(self.fc1(concat))
        y2 = F.relu(self.fc2(y1))
        y = self.fc3(y2)
        return y

# LNN dynamics model
class lnn(torch.nn.Module):
    def __init__(self, env_name, n, obs_size, action_size, dt, a_zeros):
        super(lnn, self).__init__()
        self.env_name = env_name
        self.dt = dt
        self.n = n

        input_size = obs_size - self.n
        out_L = int(self.n*(self.n+1)/2)
        self.fc1_L = torch.nn.Linear(input_size, 64)
        self.fc2_L = torch.nn.Linear(64, 64)
        self.fc3_L = torch.nn.Linear(64, out_L)
  
        self.fc1_V = torch.nn.Linear(input_size, 64)
        self.fc2_V = torch.nn.Linear(64, 64)
        self.fc3_V = torch.nn.Linear(64, 1)

        self.a_zeros = a_zeros

    def trig_transform_q(self, q):
        return torch.column_stack((torch.cos(q[:,0]),torch.sin(q[:,0])))
        
    def inverse_trig_transform_model(self, x):
        return torch.cat((torch.atan2(x[:,1],x[:,0]).unsqueeze(1),x[:,2:]),1)

    def compute_L(self, q):
        y1_L = F.softplus(self.fc1_L(q))
        y2_L = F.softplus(self.fc2_L(y1_L))
        y_L = self.fc3_L(y2_L)
        L = y_L.unsqueeze(1)
        return L

    def get_A(self, a):
        return a

    def get_L(self, q):
        trig_q = self.trig_transform_q(q)
        L = self.compute_L(trig_q)         
        return L.sum(0), L

    def get_V(self, q):
        trig_q = self.trig_transform_q(q)
        y1_V = F.softplus(self.fc1_V(trig_q))
        y2_V = F.softplus(self.fc2_V(y1_V))
        V = self.fc3_V(y2_V).squeeze()
        return V.sum()

    def get_acc(self, q, qdot, a):
        dL_dq, L = jacrev(self.get_L, has_aux=True)(q)
        term_1 = torch.einsum('blk,bijk->bijl', L, dL_dq.permute(2,3,0,1))
        dM_dq = term_1 + term_1.transpose(2,3)
        c = torch.einsum('bjik,bk,bj->bi', dM_dq, qdot, qdot) - 0.5 * torch.einsum('bikj,bk,bj->bi', dM_dq, qdot, qdot)        
        Minv = torch.cholesky_inverse(L)
        dV_dq = 0 if self.env_name == "reacher" else jacrev(self.get_V)(q)
        qddot = torch.matmul(Minv,(self.get_A(a)-c-dV_dq).unsqueeze(2)).squeeze(2)
        return qddot                                                                                                                                                                                                                                                                                                                 
                                                                                                                                                                                                           
    def derivs(self, s, a):
        q, qdot = s[:,:self.n], s[:,self.n:]
        qddot = self.get_acc(q, qdot, a)
        return torch.cat((qdot,qddot),dim=1)                                                                                                                                                               

    def rk2(self, s, a):                                                                                                                                                                                   
        alpha = 2.0/3.0 # Ralston's method                                                                                                                                                                 
        k1 = self.derivs(s, a)                                                                                                                                                                             
        k2 = self.derivs(s + alpha * self.dt * k1, a)                                                                                                                                                      
        s_1 = s + self.dt * ((1.0 - 1.0/(2.0*alpha))*k1 + (1.0/(2.0*alpha))*k2)                                                                                                                            
        return s_1

    def forward(self, o, a):
        s_1 = self.rk2(self.inverse_trig_transform_model(o), a)
        o_1 = torch.cat((self.trig_transform_q(s_1[:,:self.n]),s_1[:,self.n:]),1)
        return o_1