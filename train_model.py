import os
import argparse

import random
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from models import ReplayBuffer, dnn, lnn
from mppi import MPPI

import gymnasium

class ModelTrainer:
    def __init__(self, env, arglist):
        
        self.arglist = arglist

        random.seed(self.arglist.seed)
        np.random.seed(self.arglist.seed)
        torch.manual_seed(self.arglist.seed)

        self.env = env
        self.device = torch.device("cuda:0")

        self.name = "pendulum"
        self.obs_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]
        self.n = 1
        self.a_scale = torch.tensor([2.0],
                                    dtype=torch.double,
                                    device=self.device)

        

        path = "./log/"+self.name+"/mbrl_"+self.arglist.model
        self.exp_dir = os.path.join(path, "seed_"+str(self.arglist.seed))
        self.model_dir = os.path.join(self.exp_dir, "models")
        self.tensorboard_dir = os.path.join(self.exp_dir, "tensorboard")

        if self.arglist.mode == "train":
            
            if self.arglist.model == "lnn":
                if self.action_size < self.n:
                    a_zeros = torch.zeros(self.arglist.batch_size,
                                          self.n-self.action_size, 
                                          device=self.device)
                else:
                    a_zeros = None
                self.transition_model = lnn(self.name, 
                                            self.n, 
                                            self.obs_size, 
                                            self.action_size, 
                                            self.env.dt, 
                                            a_zeros).double().to(self.device)
            
            elif self.arglist.model == "dnn":
                self.transition_model = dnn(self.obs_size, 
                                            self.action_size).double().to(self.device)
           
            self.transition_loss_fn = torch.nn.L1Loss()

            if self.arglist.resume:
                checkpoint = torch.load(os.path.join(self.model_dir,"emergency.ckpt"))
                self.start_episode = checkpoint['episode'] + 1
                self.transition_model.load_state_dict(checkpoint['transition_model'])
                self.replay_buffer = checkpoint['replay_buffer']
            else: 
                self.start_episode = 0
                self.replay_buffer = ReplayBuffer(self.arglist.replay_size, self.device)
                if os.path.exists(path):
                    pass            
                else:
                    os.makedirs(path)
                os.mkdir(self.exp_dir)
                os.mkdir(os.path.join(self.tensorboard_dir))
                os.mkdir(self.model_dir)

            self.transition_optimizer = torch.optim.AdamW(self.transition_model.parameters(), 
                                                          lr=self.arglist.lr)

            if self.arglist.resume:
                self.transition_optimizer.load_state_dict(checkpoint['transition_optimizer'])
                print("Done loading checkpoint ...")
        
    def save_checkpoint(self, name):
        checkpoint = {'transition_model': self.transition_model.state_dict()}
        torch.save(checkpoint, os.path.join(self.model_dir, name))

    def save_emergency_checkpoint(self, episode):
        checkpoint = {'episode' : episode,
                      'transition_model' : self.transition_model.state_dict(),
                      'transition_optimizer': self.transition_optimizer.state_dict(),
                      'replay_buffer' : self.replay_buffer}
        torch.save(checkpoint, os.path.join(self.model_dir, "emergency.ckpt"))

    def angle_normalize(self, x):
        return ((x + torch.pi) % (2 * torch.pi)) - torch.pi
    
    def stage_cost(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        theta = state[:, 0]
        theta_dt = state[:, 1]
        # u = action[:, 0]
        cost = self.angle_normalize(theta) ** 2 + 0.1 * theta_dt**2
        return cost

    def terminal_cost(self, state: torch.Tensor) -> torch.Tensor:
        theta = state[:, 0]
        theta_dt = state[:, 1]
        cost = self.angle_normalize(theta) ** 2 + 0.1 * theta_dt**2
        return cost
    
    def train(self):
        writer = SummaryWriter(log_dir=self.tensorboard_dir)
        if not self.arglist.resume:
            # Initialize replay buffer with K random episodes
            for k_episode in range(self.arglist.K):
                o,_ = self.env.reset()    
                o_tensor = torch.tensor(o, device=self.device)
                ep_r = 0
                for i in range(self.arglist.T):
                    a = np.random.uniform(-2.0, 2.0, size=self.action_size)
                    o_1, r, done, _, _ = self.env.step(a)
                    a_tensor = torch.tensor(a, device=self.device)
                    o_1_tensor = torch.tensor(o_1, device=self.device)
                    r_tensor = torch.tensor(r, device=self.device)
                    self.replay_buffer.push(o_tensor, a_tensor, r_tensor, o_1_tensor)
                    ep_r += r
                    o_tensor = o_1_tensor
                    if done:
                        break
                    

            print("Done initialization ...")
        print("Started training ...")

        for episode in range(self.arglist.episodes):
            transition_loss_list = []
            transition_grad_list = []
            for model_batches in range(self.arglist.model_batches):
                O, A, R, O_1 = self.replay_buffer.sample_transitions(self.arglist.batch_size)
                # Dynamics learning
                O_1_pred = self.transition_model(O, A*self.a_scale)
                transition_loss = self.transition_loss_fn(O_1_pred, O_1)
                self.transition_optimizer.zero_grad()
                transition_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.transition_model.parameters(), self.arglist.clip_term)
                self.transition_optimizer.step()
                transition_loss_list.append(transition_loss.item())
                transition_grad = []
                for param in self.transition_model.parameters():
                    if param.grad is not None:
                        transition_grad.append(param.grad.flatten())
                transition_grad_list.append(torch.norm(torch.cat(transition_grad)).item())
            # Log training
            writer.add_scalar('transition_loss', np.mean(transition_loss_list), episode)
            writer.add_scalar('transition_grad',np.mean(transition_grad_list), episode)

            # Interact with the environment and add transition to replay buffer
            # Initialize mppi solver
            with torch.no_grad():
                print("initialize solver: ", episode)
                solver = MPPI(
                    horizon=20,
                    num_samples=1000,
                    dim_state=2,
                    dim_control=1,
                    dynamics_model=self.transition_model,
                    dynamics=None,
                    stage_cost=self.stage_cost,
                    terminal_cost=self.terminal_cost,
                    u_min=torch.tensor([-2.0]),
                    u_max=torch.tensor([2.0]),
                    sigmas=torch.tensor([1.0]),
                    lambda_=1.0,
                )
                obs, _ = self.env.reset()
                o_tensor = torch.tensor(obs, device=self.device)
                for i in range(50):
                    state = self.env.unwrapped.state.copy()

                    # solve
                    action_seq, state_seq = solver.forward(state=state)
                    action_seq_np = action_seq.detach().cpu().numpy()
                    a_tensor = action_seq[0, :]
                    
                    # update simulator
                    observation, reward, terminated, truncated, info = self.env.step(action_seq_np[0, :])
                    r_tensor = torch.tensor(reward, device=self.device)
                    o_1_tensor = torch.tensor(observation, device=self.device)
                    
                    # add to replay buffer
                    self.replay_buffer.push(o_tensor, a_tensor, r_tensor, o_1_tensor)
                    o_tensor = o_1_tensor

            if episode % self.arglist.eval_every == 0 or episode == self.arglist.episodes-1:
                try:
                    # Evaluate agent performance
                    self.save_checkpoint(str(episode)+".ckpt")
                except:
                    print("episode",episode,"got nan during eval")
                # if (episode % 25 == 0 or episode == self.arglist.episodes-1) and episode > self.start_episode:
                #     self.save_emergency_checkpoint(episode)
                #     break

def parse_args():
    parser = argparse.ArgumentParser("Model-Based Reinforcement Learning")
    # Common settings
    parser.add_argument("--mode", type=str, default="train", help="train or eval")
    parser.add_argument("--episodes", type=int, default=100, help="number of episodes to run experiment for")
    parser.add_argument("--seed", type=int, default=1, help="seed")
    # Core training parameters
    parser.add_argument("--resume", action="store_true", default=False, help="continue training from checkpoint")
    parser.add_argument("--model", type=str, default="dnn", help="lnn / dnn")
    parser.add_argument("--T", type=int, default=16, help="imagination horizon")
    parser.add_argument("--K", type=int, default=10, help="init replay buffer with K random episodes")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--clip-term", type=float, default=100, help="gradient clipping norm")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--Lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size for model learning, behaviour learning")
    parser.add_argument("--model-batches", type=int, default=int(1e4), help="model batches per episode")
    parser.add_argument("--behaviour-batches", type=int, default=int(1e3), help="behaviour batches per episode")
    parser.add_argument("--replay-size", type=int, default=int(1e5), help="replay buffer size")
    parser.add_argument("--eval-every", type=int, default=5, help="eval every _ episodes during training")
    parser.add_argument("--eval-over", type=int, default=50, help="each time eval over _ episodes")
    # Eval settings
    parser.add_argument("--checkpoint", type=str, default="", help="path to checkpoint")
    parser.add_argument("--render", action="store_true", default=False, help="render")
    return parser.parse_args()

if __name__ == '__main__':
    arglist = parse_args()
    env = gymnasium.make("Pendulum-v1")
    model_trainer = ModelTrainer(env, arglist)
    model_trainer.train()
                    


