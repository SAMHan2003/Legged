# /home/sam/Legged/deploy/check_action_order.py
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import numpy as np
import torch

args = get_args()
args.num_envs = 1
args.headless = True

env, env_cfg = task_registry.make_env(name=args.task, args=args)
env.reset()

print("=" * 60)
print("env.cfg.asset.dof_names (policy期望输出的顺序!):")
for i, n in enumerate(env.cfg.asset.dof_names):
    print(f"  action[{i}] = {n}")
print()
print("URDF解析得到的dof顺序 (env.simulator内部用的):")
# 这个顺序可以从dof_indices反推
print(f"  dof_indices = {env.dof_indices if hasattr(env, 'dof_indices') else 'N/A'}")

# 实验性：给action[0]=1,其他=0,看哪个关节动了
actions = torch.zeros(1, env.num_actions, device=env.device)
actions[0, 0] = 1.0   # 只有第0个action=1

# 记录前dof_pos
env.reset()
dof_pos_before = env.simulator.dof_pos[0].cpu().numpy().copy()

# step 10次
for _ in range(10):
    env.step(actions)

dof_pos_after = env.simulator.dof_pos[0].cpu().numpy()
diff = dof_pos_after - dof_pos_before

print()
print("Action[0]=1.0 测试 (其他action=0):")
print("dof_pos 变化量 (按simulator内部顺序):")
# 找变化最大的
for i, d in enumerate(diff):
    print(f"  dof[{i}] diff = {d:+.4f}")
print()
max_idx = np.argmax(np.abs(diff))
print(f"变化最大的是 dof[{max_idx}], 变化量={diff[max_idx]:.4f}")
print(f"这说明 action[0] 实际对应 simulator 的 dof[{max_idx}]")