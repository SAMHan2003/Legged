"""一次性运行,把训练env的所有真实参数dump到yaml,供部署脚本使用"""
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import numpy as np
import yaml

args = get_args()
args.num_envs = 1
args.headless = True

env, env_cfg = task_registry.make_env(name=args.task, args=args)

params = {
    "num_obs": int(env.num_obs),
    "num_actions": int(env.num_actions),
}

def try_add(key, getter):
    try:
        params[key] = getter()
    except Exception as e:
        print(f"⚠️  Skipping {key}: {e}")

try_add("commands_scale", lambda: env.commands_scale.cpu().numpy().tolist())
try_add("obs_scales_lin_vel", lambda: float(env.obs_scales.lin_vel))
try_add("obs_scales_ang_vel", lambda: float(env.obs_scales.ang_vel))
try_add("obs_scales_dof_pos", lambda: float(env.obs_scales.dof_pos))
try_add("obs_scales_dof_vel", lambda: float(env.obs_scales.dof_vel))
try_add("action_scale", lambda: float(env.cfg.control.action_scale))
try_add("stiffness", lambda: dict(env.cfg.control.stiffness))
try_add("damping", lambda: dict(env.cfg.control.damping))
try_add("decimation", lambda: int(env.cfg.control.decimation))
try_add("sim_dt", lambda: float(env.cfg.sim.dt))
try_add("clip_actions", lambda: float(env.cfg.normalization.clip_actions))
try_add("clip_observations", lambda: float(env.cfg.normalization.clip_observations))
try_add("dof_names_from_cfg", lambda: list(env.cfg.asset.dof_names))
try_add("default_dof_pos", lambda: env.default_dof_pos[0].cpu().numpy().tolist() 
        if hasattr(env.default_dof_pos, 'dim') and env.default_dof_pos.dim() > 1 
        else env.default_dof_pos.cpu().numpy().tolist())
try_add("p_gains", lambda: env.p_gains.cpu().numpy().tolist() 
        if env.p_gains.dim() == 1 else env.p_gains[0].cpu().numpy().tolist())
try_add("d_gains", lambda: env.d_gains.cpu().numpy().tolist()
        if env.d_gains.dim() == 1 else env.d_gains[0].cpu().numpy().tolist())

env.reset()
try_add("obs_sample_first_frame", lambda: env.obs_buf[0].cpu().numpy().tolist())

out_path = "/home/sam/Legged/deploy/configs/go2_train_params.yaml"
with open(out_path, "w") as f:
    yaml.dump(params, f, default_flow_style=False, sort_keys=False)

print(f"\n✓ Saved training params to {out_path}\n")
print("=" * 60)
print("KEY VALUES (这些决定部署能否成功):")
print("=" * 60)
for key in ['num_obs', 'action_scale', 'commands_scale', 
            'obs_scales_lin_vel', 'obs_scales_ang_vel',
            'obs_scales_dof_pos', 'obs_scales_dof_vel',
            'decimation', 'sim_dt',
            'p_gains', 'd_gains', 
            'default_dof_pos', 'dof_names_from_cfg']:
    if key in params:
        print(f"  {key}: {params[key]}")