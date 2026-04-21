"""
打印训练env里action[i]实际对应哪个关节。
用Isaac Gym加载env，检查dof_names属性。
"""
import isaacgym  # 必须先import
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

args = get_args()
args.task = "go2_ts_depth"    # 改成你训练的任务名
args.num_envs = 1
args.headless = True

env, env_cfg = task_registry.make_env(name=args.task, args=args)

print("=" * 60)
print("dof_names (这就是action[0..11]对应的关节顺序):")
for i, name in enumerate(env.dof_names):
    print(f"  action[{i}] -> {name}")
print("=" * 60)
print("default_dof_pos:")
print(env.default_dof_pos[0].cpu().numpy())