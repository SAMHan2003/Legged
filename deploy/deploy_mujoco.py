import os
LEGGED_GYM_ROOT_DIR = os.path.join(os.path.dirname(__file__), "../LeggedGym-Ex")

import time
import argparse
import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml


# ============ 调试开关 ============
POLICY_DISABLED = False   # True=只做PD保持default, False=跑policy
DEBUG_PRINT = True       # True=每0.2s打印一次状态
# ==================================


def get_gravity_orientation(quaternion):
    """Mujoco quat convention: (w, x, y, z)"""
    qw, qx, qy, qz = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    g = np.zeros(3)
    g[0] = 2 * (-qz * qx + qw * qy)
    g[1] = -2 * (qz * qy + qw * qx)
    g[2] = 1 - 2 * (qw * qw + qz * qz)
    return g


# Policy / Legged Gym 的关节顺序 (从 env.cfg.asset.dof_names 确认)
POLICY_JOINT_NAMES = [
    'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
    'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
    'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
    'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
]


def build_mujoco_mappings(model):
    """自动从mujoco model构造两个映射表,避免手写错误."""
    # 从关节名找policy->qpos的映射
    qpos_adr_from_policy = []
    for jname in POLICY_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        assert jid >= 0, f"Joint {jname} not found in mujoco model!"
        qpos_adr_from_policy.append(model.jnt_qposadr[jid])
        
    # dof address (for qvel)
    dof_adr_from_policy = []
    for jname in POLICY_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        dof_adr_from_policy.append(model.jnt_dofadr[jid])

    # 从actuator名找policy->ctrl的映射
    # mujoco的actuator名一般是 "FL_hip", "FL_thigh", "FL_calf" (没有_joint后缀)
    ctrl_from_policy = []
    for jname in POLICY_JOINT_NAMES:
        # 把 FL_hip_joint -> FL_hip
        act_name = jname.replace('_joint', '')
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
        assert aid >= 0, f"Actuator {act_name} not found!"
        ctrl_from_policy.append(aid)
    
    return (np.array(qpos_adr_from_policy),
            np.array(dof_adr_from_policy),
            np.array(ctrl_from_policy))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    args = parser.parse_args()

    with open(f"configs/{args.config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    xml_path    = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

    sim_duration       = config["simulation_duration"]
    sim_dt             = config["simulation_dt"]
    control_decimation = config["control_decimation"]

    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)

    ang_vel_scale = config["ang_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    action_scale  = config["action_scale"]
    cmd_scale     = np.array(config["cmd_scale"], dtype=np.float32)

    num_actions = config["num_actions"]
    num_obs     = config["num_obs"]
    cmd = np.array(config["cmd_init"], dtype=np.float32)

    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0

    # Load mujoco
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = sim_dt

    # Build mappings using joint NAMES (不再手写,100%正确)
    qpos_adr_from_policy, dof_adr_from_policy, ctrl_from_policy = build_mujoco_mappings(m)
    
    print("="*60)
    print("Joint mapping (policy index -> mujoco):")
    for i, jname in enumerate(POLICY_JOINT_NAMES):
        print(f"  policy[{i:2d}] {jname:15s} -> qpos_adr={qpos_adr_from_policy[i]}, "
              f"dof_adr={dof_adr_from_policy[i]}, ctrl={ctrl_from_policy[i]}")
    print("="*60)

    # Initialize state
    d.qpos[2] = 0.42
    for i in range(12):
        d.qpos[qpos_adr_from_policy[i]] = default_angles[i]
    mujoco.mj_forward(m, d)
    
    print(f"Initial qpos[7:19] = {d.qpos[7:19]}")
    print(f"Expected (per policy order, but mujoco qpos order): {default_angles}")
    print("="*60)

    # Load policy
    policy = torch.jit.load(policy_path)
    print(f"Loaded policy: {policy_path}")
    if POLICY_DISABLED:
        print("⚠️  POLICY DISABLED - running PD-only mode")
    print("="*60)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        debug_step = int(0.2 / sim_dt)   # 每0.2秒打印一次

        while viewer.is_running() and time.time() - start < sim_duration:
            step_start = time.time()

            # Read state via index mapping (policy order)
            q_policy  = np.array([d.qpos[qpos_adr_from_policy[i]] for i in range(12)], dtype=np.float32)
            dq_policy = np.array([d.qvel[dof_adr_from_policy[i]]  for i in range(12)], dtype=np.float32)

            # PD in policy order
            tau_policy = (target_dof_pos - q_policy) * kps + (0.0 - dq_policy) * kds
            
            # Write to ctrl via index mapping
            for i in range(12):
                d.ctrl[ctrl_from_policy[i]] = tau_policy[i]

            mujoco.mj_step(m, d)
            counter += 1

            # Policy inference
            if not POLICY_DISABLED and counter % control_decimation == 0:
                quat  = d.qpos[3:7].copy()
                omega = d.qvel[3:6].astype(np.float32)
                qj_obs  = (q_policy - default_angles) * dof_pos_scale
                dqj_obs = dq_policy * dof_vel_scale
                gravity = get_gravity_orientation(quat)
                omega_obs = omega * ang_vel_scale

                obs[0:3]   = cmd * cmd_scale
                obs[3:6]   = gravity
                obs[6:9]   = omega_obs
                obs[9:21]  = qj_obs
                obs[21:33] = dqj_obs
                obs[33:45] = action

                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action = policy(obs_tensor).detach().numpy().squeeze().astype(np.float32)
                action = np.clip(action, -100.0, 100.0)
                target_dof_pos = action * action_scale + default_angles

            # Debug print
            if DEBUG_PRINT and counter % debug_step == 1:
                print(f"\n--- step {counter} t={counter*sim_dt:.2f}s ---")
                print(f"  base_z={d.qpos[2]:.3f}  quat={d.qpos[3:7].round(3)}")
                print(f"  q_policy  (FL)={q_policy[0:3].round(3)}  (FR)={q_policy[3:6].round(3)}")
                print(f"  q_policy  (RL)={q_policy[6:9].round(3)}  (RR)={q_policy[9:12].round(3)}")
                print(f"  tau_policy(FL)={tau_policy[0:3].round(2)}  (FR)={tau_policy[3:6].round(2)}")
                print(f"  tau_policy(RL)={tau_policy[6:9].round(2)}  (RR)={tau_policy[9:12].round(2)}")

            viewer.sync()
            t_sleep = m.opt.timestep - (time.time() - step_start)
            if t_sleep > 0:
                time.sleep(t_sleep)