"""
Unitree Go2 real/sim deployment script.

Same script works for:
  - Real Go2   :  sudo python deploy_real_go2.py eth0 go2.yaml
  - unitree_mujoco sim :  python deploy_real_go2.py lo   go2.yaml
    (start unitree_mujoco first, it publishes on loopback)

The policy / obs construction matches what you already validated in
deploy_mujoco.py (cmd -> gravity -> ang_vel -> q -> dq -> action).

Safety flow:
    1) Zero-torque wait        -> press ENTER
    2) Smooth move to default  -> press ENTER
    3) Run policy              -> Ctrl+C to enter damping state and exit
"""

import os
import sys
import time
import yaml
import argparse
import threading
import numpy as np
import torch

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient


# ================================================================
# Constants
# ================================================================

HW_JOINT_NAMES = [
    'FR_hip', 'FR_thigh', 'FR_calf',   # 0,1,2
    'FL_hip', 'FL_thigh', 'FL_calf',   # 3,4,5
    'RR_hip', 'RR_thigh', 'RR_calf',   # 6,7,8
    'RL_hip', 'RL_thigh', 'RL_calf',   # 9,10,11
]

POLICY_ORDERS = {
    # mujoco script's POLICY_JOINT_NAMES (FR first)
    "FR_FL_RR_RL": [
        'FR_hip', 'FR_thigh', 'FR_calf',
        'FL_hip', 'FL_thigh', 'FL_calf',
        'RR_hip', 'RR_thigh', 'RR_calf',
        'RL_hip', 'RL_thigh', 'RL_calf',
    ],
    # yaml comment order (FL first)
    "FL_FR_RL_RR": [
        'FL_hip', 'FL_thigh', 'FL_calf',
        'FR_hip', 'FR_thigh', 'FR_calf',
        'RL_hip', 'RL_thigh', 'RL_calf',
        'RR_hip', 'RR_thigh', 'RR_calf',
    ],
}


def build_policy_to_hw_mapping(policy_order_name: str):
    """
    Return arr such that  hw_index = arr[policy_index].

    Example: if policy_order is FR_FL_RR_RL it happens to match hardware,
    so arr = [0,1,2,3,4,5,6,7,8,9,10,11].  If policy is FL_FR_RL_RR,
    policy[0] ('FL_hip') needs hw index 3, etc.
    """
    if policy_order_name not in POLICY_ORDERS:
        raise ValueError(
            f"Unknown policy_joint_order '{policy_order_name}'. "
            f"Must be one of {list(POLICY_ORDERS.keys())}"
        )
    policy_names = POLICY_ORDERS[policy_order_name]
    mapping = np.array(
        [HW_JOINT_NAMES.index(n) for n in policy_names],
        dtype=np.int32,
    )
    return mapping


# ================================================================
# Math helpers  (identical to deploy_mujoco.py so obs stay bit-identical)
# ================================================================

def get_gravity_orientation(quaternion):
    """IMU quat convention on Go2 SDK: (w, x, y, z)."""
    qw, qx, qy, qz = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    g = np.zeros(3, dtype=np.float32)
    g[0] = 2 * (-qz * qx + qw * qy)
    g[1] = -2 * (qz * qy + qw * qx)
    g[2] = 1 - 2 * (qw * qw + qz * qz)
    return g


# ================================================================
# Low-level command helpers for Go2
# ================================================================

# Go2 motor modes.  0x01 = PMSM servo mode (position+velocity+torque control).
# 0x00 = damping / disabled.
GO2_MODE_SERVO = 0x01

LOWCMD_TOPIC   = "rt/lowcmd"
LOWSTATE_TOPIC = "rt/lowstate"


def _init_cmd_struct(cmd: LowCmdGo):
    """Fill the fixed header bytes of a go LowCmd message."""
    cmd.head[0] = 0xFE
    cmd.head[1] = 0xEF
    cmd.level_flag = 0xFF        # low-level control
    cmd.gpio = 0
    for i in range(20):
        cmd.motor_cmd[i].mode = GO2_MODE_SERVO
        cmd.motor_cmd[i].q    = 0.0
        cmd.motor_cmd[i].dq   = 0.0
        cmd.motor_cmd[i].kp   = 0.0
        cmd.motor_cmd[i].kd   = 0.0
        cmd.motor_cmd[i].tau  = 0.0


def _fill_zero_torque(cmd: LowCmdGo):
    """All gains zero, all targets zero -> motors produce no torque."""
    for i in range(12):
        cmd.motor_cmd[i].q   = 0.0
        cmd.motor_cmd[i].dq  = 0.0
        cmd.motor_cmd[i].kp  = 0.0
        cmd.motor_cmd[i].kd  = 0.0
        cmd.motor_cmd[i].tau = 0.0


def _fill_damping(cmd: LowCmdGo, kd: float = 3.0):
    """Emergency stop: zero targets, only velocity damping.  Robot slumps safely."""
    for i in range(12):
        cmd.motor_cmd[i].q   = 0.0
        cmd.motor_cmd[i].dq  = 0.0
        cmd.motor_cmd[i].kp  = 0.0
        cmd.motor_cmd[i].kd  = kd
        cmd.motor_cmd[i].tau = 0.0


# ================================================================
# Controller
# ================================================================

class Go2Deployer:
    def __init__(self, cfg: dict):
        # ---------------- config ----------------
        self.cfg = cfg

        self.control_dt = float(cfg["simulation_dt"]) * int(cfg["control_decimation"])
        # e.g. 0.005 * 4 = 0.02  -> 50 Hz policy rate
        self.num_actions = int(cfg["num_actions"])
        self.num_obs     = int(cfg["num_obs"])

        self.kps            = np.array(cfg["kps"],            dtype=np.float32)
        self.kds            = np.array(cfg["kds"],            dtype=np.float32)
        self.default_angles = np.array(cfg["default_angles"], dtype=np.float32)

        self.ang_vel_scale = float(cfg["ang_vel_scale"])
        self.dof_pos_scale = float(cfg["dof_pos_scale"])
        self.dof_vel_scale = float(cfg["dof_vel_scale"])
        self.action_scale  = float(cfg["action_scale"])
        self.cmd_scale     = np.array(cfg["cmd_scale"], dtype=np.float32)
        self.cmd           = np.array(cfg["cmd_init"],  dtype=np.float32)

        # policy joint order switch (the whole point of this script)
        order_name = cfg.get("policy_joint_order", "FR_FL_RR_RL")
        self.policy_to_hw = build_policy_to_hw_mapping(order_name)
        print(f"[cfg] policy_joint_order = {order_name}")
        print(f"[cfg] policy->hw mapping = {self.policy_to_hw.tolist()}")

        # ---------------- policy ----------------
        # Resolve {LEGGED_GYM_ROOT_DIR}.  Priority:
        #   1) env var LEGGED_GYM_ROOT_DIR if set
        #   2) <this_script_dir>/../LeggedGym-Ex   (same default as deploy_mujoco.py)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_root = os.path.normpath(os.path.join(script_dir, "../LeggedGym-Ex"))
        legged_gym_root = os.environ.get("LEGGED_GYM_ROOT_DIR", default_root)
        print(f"[cfg] LEGGED_GYM_ROOT_DIR = {legged_gym_root}")

        policy_path = cfg["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", legged_gym_root)
        print(f"[cfg] loading policy: {policy_path}")
        if not os.path.isfile(policy_path):
            raise FileNotFoundError(
                f"Policy file not found: {policy_path}\n"
                f"Either:\n"
                f"  - set env var LEGGED_GYM_ROOT_DIR to point at your LeggedGym-Ex root, or\n"
                f"  - edit policy_path in the yaml to an absolute path."
            )
        self.policy = torch.jit.load(policy_path)
        self.policy.eval()

        self.action         = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos = self.default_angles.copy()
        self.obs            = np.zeros(self.num_obs,     dtype=np.float32)
        self.counter        = 0

        self.low_cmd   = unitree_go_msg_dds__LowCmd_()
        self.low_state = unitree_go_msg_dds__LowState_()
        _init_cmd_struct(self.low_cmd)
        self._crc = CRC()

        self._state_lock = threading.Lock()
        self._got_first_state = False

        self.pub = ChannelPublisher(LOWCMD_TOPIC, LowCmdGo)
        self.pub.Init()

        self.sub = ChannelSubscriber(LOWSTATE_TOPIC, LowStateGo)
        self.sub.Init(self._on_low_state, 10)

        try:
            print("[io] initializing SportClient / MotionSwitcherClient ...")
            self._sc = SportClient()
            self._sc.SetTimeout(5.0)
            self._sc.Init()

            self._msc = MotionSwitcherClient()
            self._msc.SetTimeout(5.0)
            self._msc.Init()

            print("[io] releasing any high-level mode that's holding rt/lowcmd ...")
            status, result = self._msc.CheckMode()
            tries = 0
            while result.get('name'):
                print(f"[io]   current mode = {result['name']}, releasing ...")
                self._sc.StandDown()
                self._msc.ReleaseMode()
                time.sleep(1.0)
                status, result = self._msc.CheckMode()
                tries += 1
                if tries > 5:
                    print("[io]   warning: still holding mode after 5 tries, continuing anyway")
                    break
            print("[io] high-level mode released, we own rt/lowcmd now.")
        except Exception as e:
            print(f"[io] SportClient/MotionSwitcher skipped ({e}). "
                  f"This is expected in unitree_mujoco simulation.")

    # ---------- callbacks ----------

    def _on_low_state(self, msg: LowStateGo):
        with self._state_lock:
            self.low_state = msg
            self._got_first_state = True

    # ---------- helpers ----------

    def send_cmd(self):
        self.low_cmd.crc = self._crc.Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)

    def wait_for_low_state(self, timeout: float = 5.0):
        """Wait until at least one LowState message has been received.

        Note: we deliberately do NOT check `tick != 0`.  The Python version of
        unitree_mujoco never updates the tick field, so that check hangs forever
        in simulation.  Real Go2 firmware does set tick, but "got any message"
        is already sufficient proof of connectivity.
        """
        t0 = time.time()
        while True:
            with self._state_lock:
                ok = self._got_first_state
            if ok:
                break
            if time.time() - t0 > timeout:
                raise RuntimeError(
                    "No LowState received.  Check network interface, "
                    "and that unitree_mujoco or the real robot is running."
                )
            time.sleep(self.control_dt)
        with self._state_lock:
            tick = self.low_state.tick
        print(f"[io] connected to robot, LowState flowing.  (first tick={tick})")


    def read_joint_state_policy_order(self):
        """Return (q, dq) of length 12 in policy ordering."""
        q  = np.zeros(12, dtype=np.float32)
        dq = np.zeros(12, dtype=np.float32)
        with self._state_lock:
            ms = self.low_state.motor_state
            for p in range(12):
                h = int(self.policy_to_hw[p])
                q[p]  = ms[h].q
                dq[p] = ms[h].dq
        return q, dq

    def read_imu(self):
        with self._state_lock:
            quat = np.array(self.low_state.imu_state.quaternion, dtype=np.float32)  # (w,x,y,z)
            gyro = np.array(self.low_state.imu_state.gyroscope,  dtype=np.float32)  # rad/s
        return quat, gyro

    # ---------- write in POLICY ----------

    def write_pd_cmd_policy_order(self, q_target, kp, kd):
        """q_target, kp, kd are length-12 arrays in policy order."""
        for p in range(12):
            h = int(self.policy_to_hw[p])
            m = self.low_cmd.motor_cmd[h]
            m.mode = GO2_MODE_SERVO
            m.q    = float(q_target[p])
            m.dq   = 0.0
            m.kp   = float(kp[p])
            m.kd   = float(kd[p])
            m.tau  = 0.0


    def phase_zero_torque_until_enter(self):
        print("\n[phase 1] zero-torque.  Robot should be limp.")
        print("          Press ENTER to move to default stance, Ctrl+C to abort.")
        enter_pressed = {"v": False}

        def wait_enter():
            try:
                input()
            except EOFError:
                pass
            enter_pressed["v"] = True

        t = threading.Thread(target=wait_enter, daemon=True)
        t.start()

        while not enter_pressed["v"]:
            _fill_zero_torque(self.low_cmd)
            self.send_cmd()
            time.sleep(self.control_dt)

    def phase_move_to_default(self, total_time: float = 2.0):
        print(f"\n[phase 2] moving to default stance over {total_time:.1f}s ...")
        # snapshot starting joint positions (policy order)
        q_start, _ = self.read_joint_state_policy_order()
        q_target   = self.default_angles.copy()

        n_steps = max(1, int(total_time / self.control_dt))
        for i in range(n_steps):
            alpha = (i + 1) / n_steps
            q_cmd = (1.0 - alpha) * q_start + alpha * q_target
            self.write_pd_cmd_policy_order(q_cmd, self.kps, self.kds)
            self.send_cmd()
            time.sleep(self.control_dt)
        print("[phase 2] reached default stance.")

    def phase_hold_default_until_enter(self):
        print("\n[phase 3] holding default stance.")
        print("          Press ENTER to start the policy, Ctrl+C to abort.")
        enter_pressed = {"v": False}

        def wait_enter():
            try:
                input()
            except EOFError:
                pass
            enter_pressed["v"] = True

        t = threading.Thread(target=wait_enter, daemon=True)
        t.start()

        while not enter_pressed["v"]:
            self.write_pd_cmd_policy_order(self.default_angles, self.kps, self.kds)
            self.send_cmd()
            time.sleep(self.control_dt)

    def phase_run_policy(self, debug_print: bool = True):
        print("\n[phase 4] running policy.  Ctrl+C to stop.")
        debug_step = max(1, int(0.5 / self.control_dt))  # ~0.5s

        try:
            while True:
                t_start = time.time()
                self.counter += 1

                q_policy, dq_policy = self.read_joint_state_policy_order()
                quat, gyro = self.read_imu()

                qj_obs    = (q_policy - self.default_angles) * self.dof_pos_scale
                dqj_obs   = dq_policy * self.dof_vel_scale
                gravity   = get_gravity_orientation(quat)
                omega_obs = gyro * self.ang_vel_scale

                self.obs[0:3]   = self.cmd * self.cmd_scale
                self.obs[3:6]   = gravity
                self.obs[6:9]   = omega_obs
                self.obs[9:21]  = qj_obs
                self.obs[21:33] = dqj_obs
                self.obs[33:45] = self.action

                with torch.no_grad():
                    obs_t = torch.from_numpy(self.obs).unsqueeze(0)
                    act_t = self.policy(obs_t)
                self.action = act_t.detach().numpy().squeeze().astype(np.float32)
                self.action = np.clip(self.action, -100.0, 100.0)

                self.target_dof_pos = self.action * self.action_scale + self.default_angles
                self.write_pd_cmd_policy_order(self.target_dof_pos, self.kps, self.kds)
                self.send_cmd()

                if debug_print and self.counter % debug_step == 1:
                    print(
                        f"[run] step={self.counter:5d} "
                        f"gravity={gravity.round(2)} "
                        f"omega={gyro.round(2)} "
                        f"q[0:3]={q_policy[0:3].round(2)} "
                        f"act[0:3]={self.action[0:3].round(2)}"
                    )

                dt_left = self.control_dt - (time.time() - t_start)
                if dt_left > 0:
                    time.sleep(dt_left)
        except KeyboardInterrupt:
            print("\n[run] KeyboardInterrupt -> entering damping state.")

    def phase_damping(self, duration: float = 1.0):
        """Final safe state: velocity damping, no torque targets."""
        print("[phase 5] damping state.")
        n = max(1, int(duration / self.control_dt))
        for _ in range(n):
            _fill_damping(self.low_cmd, kd=3.0)
            self.send_cmd()
            time.sleep(self.control_dt)
        print("[phase 5] done.  Exit.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str,
                        help="network interface. 'lo' for unitree_mujoco, "
                             "'eth0'/'enp...' for real robot.")
    parser.add_argument("config", type=str,
                        help="yaml file name under ./configs/ , e.g. go2.yaml")
    parser.add_argument("--domain", type=int, default=0,
                        help="DDS domain id. unitree_mujoco (python) default is 1, "
                             "unitree_mujoco (C++) default is 0, real robot is 0.")
    parser.add_argument("--no-debug", action="store_true",
                        help="suppress periodic debug prints")
    args = parser.parse_args()

    # resolve config path: try ./configs/<name> first, then raw path
    candidates = [
        os.path.join(os.path.dirname(__file__), "configs", args.config),
        args.config,
    ]
    config_path = next((p for p in candidates if os.path.isfile(p)), None)
    if config_path is None:
        print(f"[fatal] config not found.  Tried: {candidates}")
        sys.exit(1)
    print(f"[cfg] loading {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # bring up DDS on the chosen interface
    print(f"[dds] ChannelFactoryInitialize on '{args.net}' with domain_id={args.domain}")
    ChannelFactoryInitialize(args.domain, args.net)

    dep = Go2Deployer(cfg)
    dep.wait_for_low_state()

    try:
        dep.phase_zero_torque_until_enter()
        dep.phase_move_to_default(total_time=2.0)
        dep.phase_hold_default_until_enter()
        dep.phase_run_policy(debug_print=not args.no_debug)
    except KeyboardInterrupt:
        print("\n[main] interrupted before policy start.")
    finally:
        dep.phase_damping(duration=1.0)


if __name__ == "__main__":
    main()
