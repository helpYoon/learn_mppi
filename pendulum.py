import torch

import time
import gymnasium
import fire

from mppi import MPPI
from models import dnn, lnn

@torch.jit.script
def angle_normalize(x):
    return ((x + torch.pi) % (2 * torch.pi)) - torch.pi

def main(save_mode: bool = True, use_trained: bool = False):
    # dynamics and cost
    # @torch.jit.script
    def true_dynamics(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # dynamics from gymnasium
        th = state[:, 0].view(-1, 1)
        thdot = state[:, 1].view(-1, 1)
        g = 10
        m = 1
        l = 1
        dt = 0.05
        u = action[:, 0].view(-1, 1)
        u = torch.clamp(u, -2, 2)
        newthdot = (
            thdot
            + (-3 * g / (2 * l) * torch.sin(th + torch.pi) + 3.0 / (m * l**2) * u)
            * dt
        )
        newth = th + newthdot * dt
        newthdot = torch.clamp(newthdot, -8, 8)

        state = torch.cat((newth, newthdot), dim=1)
        return state

    # @torch.jit.script
    def stage_cost(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        theta = state[:, 0]
        theta_dt = state[:, 1]
        # u = action[:, 0]
        cost = angle_normalize(theta) ** 2 + 0.1 * theta_dt**2
        return cost

    # @torch.jit.script
    def terminal_cost(state: torch.Tensor) -> torch.Tensor:
        theta = state[:, 0]
        theta_dt = state[:, 1]
        cost = angle_normalize(theta) ** 2 + 0.1 * theta_dt**2
        return cost

    # simulator
    if save_mode:
        env = gymnasium.make("Pendulum-v1", render_mode="rgb_array")
        env = gymnasium.wrappers.RecordVideo(env=env, video_folder="")
    else:
        env = gymnasium.make("Pendulum-v1", render_mode="human")
    observation, _ = env.reset(seed=42)

    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    
    dnn_model = dnn(obs_size, action_size).double().to(torch.device("cuda:0"))
    dnn_model.load_state_dict(torch.load('dnn_49.ckpt')['transition_model'])
    # lnn_model = lnn(env_name="pendulum", n=1, 
    #                 obs_size=obs_size,
    #                 action_size= action_size, 
    #                 dt=0.05, a_zeros= None).double().to(torch.device("cuda:0"))
    # lnn_model.load_state_dict(torch.load('lnn_99.ckpt')['transition_model'])

    # solver
    solver = MPPI(
        horizon=15,
        num_samples=1000,
        dim_state=2,
        dim_control=1,
        dynamics_model=None,
        dynamics=true_dynamics,
        stage_cost=stage_cost,
        terminal_cost=terminal_cost,
        u_min=torch.tensor([-2.0]),
        u_max=torch.tensor([2.0]),
        sigmas=torch.tensor([1.0]),
        lambda_=1.0,
    )
    with torch.no_grad():
        # solver = MPPI(
        #             horizon=15,
        #             num_samples=1000,
        #             dim_state=2,
        #             dim_control=1,
        #             dynamics_model=dnn_model,
        #             dynamics=None,
        #             stage_cost=stage_cost,
        #             terminal_cost=terminal_cost,
        #             u_min=torch.tensor([-2.0]),
        #             u_max=torch.tensor([2.0]),
        #             sigmas=torch.tensor([1.0]),
        #             lambda_=1.0,
        #         )

        average_time = 0
        total_reward = 0
        for i in range(200):
            state = env.unwrapped.state.copy()
            print(state)
            # solve
            start = time.time()
            action_seq, state_seq = solver.forward(state=state)
            elipsed_time = time.time() - start
            average_time = i / (i + 1) * average_time + elipsed_time / (i + 1)

            action_seq_np = action_seq.detach().cpu().numpy()
            state_seq_np = state_seq.detach().cpu().numpy()

            # update simulator
            observation, reward, terminated, truncated, info = env.step(action_seq_np[0, :])
            total_reward += reward
            env.render()

        print("average solve time: {}".format(average_time * 1000), " [ms]")
        print("total reward: ", total_reward)
        # env.close()


if __name__ == "__main__":
    main()