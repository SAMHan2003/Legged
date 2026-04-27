"""
Unitree Go2 real/sim deployment script (PS5 JOYSTICK EDITION).

Same script works for:
  - Real Go2            :  sudo python deploy_real_go2_joystick.py eth0 go2.yaml
  - unitree_mujoco sim  :  python deploy_real_go2_joystick.py lo  go2.yaml

Control flow (PS5 DualSense pad, verified button indices):
    1) Zero-torque state       -> press  TRIANGLE (△)   to stand up
    2) Smooth move to default  -> press  CROSS (×)      to run policy
    3) Run policy              -> left/right sticks drive the robot
                                  press  SQUARE (□)     to exit to damping
                                  or Ctrl+C in the terminal

PS5 stick mapping:
    left stick Y  (push forward) -> vx       (forward/back)
    left stick X  (push left)    -> vy       (strafe)
    right stick X (push left)    -> yaw_rate (turn)

Requires: pip install pygame
"""

import os
import sys
import time
import yaml
import argparse
import threading
import numpy as np
import torch
import pygame

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
    'FR_hip', 'FR_thigh', 'FR_calf',
    'FL_hip', 'FL_thigh', 'FL_calf',
    'RR_hip', 'RR_thigh', 'RR_calf',
    'RL_hip', 'RL_thigh', 'RL_calf',
]

POLICY_ORDERS = {
    "FR_FL_RR_RL": [
        'FR_hip', 'FR_thigh', 'FR_calf',
        'FL_hip', 'FL_thigh', 'FL_calf',
        'RR_hip', 'RR_thigh', 'RR_calf',
        'RL_hip', 'RL_thigh', 'RL_calf',
    ],
    "FL_FR_RL_RR": [
        'FL_hip', 'FL_thigh', 'FL_calf',
        'FR_hip', 'FR_thigh', 'FR_calf',
        'RL_hip', 'RL_thigh', 'RL_calf',
        'RR_hip', 'RR_thigh', 'RR_calf',
    ],
}


def build_policy_to_hw_mapping(policy_order_name: str):
    if policy_order_name not in POLICY_ORDERS:
        raise ValueError(
            f"Unknown policy_joint_order '{policy_order_name}'. "
            f"Must be one of {list(POLICY_ORDERS.keys())}"
        )
    policy_names = POLICY_ORDERS[policy_order_name]
    return np.array([HW_JOINT_NAMES.index(n) for n in policy_names], dtype=np.int32)


# ================================================================
# PS5 Joystick (DualSense) controller
# ================================================================
#
# On Linux + pygame, a DualSense shows up as a standard gamepad.
# Axis indices match SDL's mapping:
#     axis 0 : Left stick X   (-1 left, +1 right)
#     axis 1 : Left stick Y   (-1 up/forward, +1 down/back)   <- note inverted
#     axis 2 : Right stick X  (-1 left, +1 right)
#     axis 3 : Right stick Y
#     axis 4 : L2 trigger     (-1 released, +1 pressed)
#     axis 5 : R2 trigger
#
# Button indices (SDL DualSense, Linux):
#     0 = Cross (×)        <- "A" on Xbox
#     1 = Circle (○)       <- "B"
#     2 = Square (□)       <- "X"
#     3 = Triangle (△)     <- "Y"
#     4 = Share / Create   <- used as "select/exit"
#     5 = PS logo
#     6 = Options          <- used as "start"
#     7 = L3 (left stick click)
#     8 = R3 (right stick click)
#     9 = L1
#    10 = R1
#   (D-pad on PS5 is typically buttons 11-14 or a hat, driver-dependent)
#
# If your distro's driver numbers things differently, run the tiny
# diagnostic script at the bottom of this file (--probe-joystick) to
# find the right indices, then edit the constants below.

class PS5Map:
    """Verified on 'Sony Interactive Entertainment Wireless Controller' via
    SDL2 on Ubuntu 20.04 (pygame 2.6.1, 6 axes / 13 buttons).

    Button layout reported by `--probe-joystick`:
        0 = Cross (X)      1 = Circle (O)
        2 = Triangle       3 = Square
        4 = L1             5 = R1
        6 = L2 (as btn)    7 = R2 (as btn)
        (OPTIONS / CREATE / PS / touchpad are somewhere in 8..12)

    Axes:
        0 = LX  (push left  -> -1)
        1 = LY  (push up    -> -1)
        3 = RX  (push left  -> -1)
        4 = RY  (push up    -> -1)
        2 = L2 trigger       5 = R2 trigger
    """
    # --- flow-control buttons (picked to be visually distinct face buttons) ---
    BTN_START_STAND = 2   # Triangle : zero-torque  -> move to default
    BTN_START_RUN   = 0   # Cross    : default pose -> run policy
    BTN_EXIT        = 3   # Square   : run policy   -> damping / exit
    # --- stick axes ---
    AXIS_LX = 0
    AXIS_LY = 1
    AXIS_RX = 3
    AXIS_RY = 4


class PS5Joystick:
    """Thin pygame wrapper giving us .lx/.ly/.rx/.ry and .button[...]."""

    def __init__(self, deadzone: float = 0.08):
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError(
                "No joystick detected. Plug in the PS5 controller (USB or BT) "
                "and make sure you can see it with `ls /dev/input/js*`."
            )
        self.js = pygame.joystick.Joystick(0)
        self.js.init()
        print(f"[joystick] opened: {self.js.get_name()}  "
              f"axes={self.js.get_numaxes()}  buttons={self.js.get_numbuttons()}")
        self.deadzone = deadzone

        self.lx = 0.0
        self.ly = 0.0
        self.rx = 0.0
        self.ry = 0.0
        # allocate a big enough button array for any layout we might see
        self.button = [0] * max(16, self.js.get_numbuttons())

    def _apply_deadzone(self, v: float) -> float:
        return 0.0 if abs(v) < self.deadzone else v

    def poll(self):
        """Pump pygame events and refresh sticks/buttons. Call once per control step."""
        pygame.event.pump()
        self.lx = self._apply_deadzone(self.js.get_axis(PS5Map.AXIS_LX))
        self.ly = self._apply_deadzone(self.js.get_axis(PS5Map.AXIS_LY))
        self.rx = self._apply_deadzone(self.js.get_axis(PS5Map.AXIS_RX))
        self.ry = self._apply_deadzone(self.js.get_axis(PS5Map.AXIS_RY))
        for i in range(self.js.get_numbuttons()):
            self.button[i] = self.js.get_button(i)


# ================================================================
# Math helpers
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

GO2_MODE_SERVO = 0x01
LOWCMD_TOPIC   = "rt/lowcmd"
LOWSTATE_TOPIC = "rt/lowstate"


def _init_cmd_struct(cmd: LowCmdGo):
    cmd.head[0] = 0xFE
    cmd.head[1] = 0xEF
    cmd.level_flag = 0xFF
    cmd.gpio = 0
    for i in range(20):
        cmd.motor_cmd[i].mode = GO2_MODE_SERVO
        cmd.motor_cmd[i].q    = 0.0
        cmd.motor_cmd[i].dq   = 0.0
        cmd.motor_cmd[i].kp   = 0.0
        cmd.motor_cmd[i].kd   = 0.0
        cmd.motor_cmd[i].tau  = 0.0


def _fill_zero_torque(cmd: LowCmdGo):
    for i in range(12):
        cmd.motor_cmd[i].q   = 0.0
        cmd.motor_cmd[i].dq  = 0.0
        cmd.motor_cmd[i].kp  = 0.0
        cmd.motor_cmd[i].kd  = 0.0
        cmd.motor_cmd[i].tau = 0.0


def _fill_damping(cmd: LowCmdGo, kd: float = 3.0):
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
    def __init__(self, cfg: dict, joystick: PS5Joystick):
        # ---------------- config ----------------
        self.cfg = cfg
        self.joy = joystick

        self.control_dt = float(cfg["simulation_dt"]) * int(cfg["control_decimation"])
        # Go2 policy: 45-dim obs, 12-dim action. Override via yaml if needed.
        self.num_actions = int(cfg.get("num_actions", 12))
        self.num_obs     = int(cfg.get("num_obs", 45))

        self.kps            = np.array(cfg["kps"],            dtype=np.float32)
        self.kds            = np.array(cfg["kds"],            dtype=np.float32)
        self.default_angles = np.array(cfg["default_angles"], dtype=np.float32)

        self.ang_vel_scale = float(cfg["ang_vel_scale"])
        self.dof_pos_scale = float(cfg["dof_pos_scale"])
        self.dof_vel_scale = float(cfg["dof_vel_scale"])
        self.action_scale  = float(cfg["action_scale"])
        self.cmd_scale     = np.array(cfg["cmd_scale"], dtype=np.float32)

        # cmd is now driven by the joystick, start at zero.
        self.cmd = np.zeros(3, dtype=np.float32)

        # Safety caps on the velocity commands (applied AFTER reading the
        # joystick). Prevents the user from saturating the policy's training
        # distribution. Tune per your training range.
        self.cmd_max = np.array(
            cfg.get("joystick_cmd_max", [0.5, 0.3, 0.5]),
            dtype=np.float32,
        )
        # Sign flips, in case your training convention is different.
        self.cmd_sign = np.array(
            cfg.get("joystick_cmd_sign", [1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

        # Joint mapping: prefer explicit `policy_to_sdk` list (original yaml
        # format), fall back to `policy_joint_order` string (named preset).
        if "policy_to_sdk" in cfg:
            self.policy_to_hw = np.array(cfg["policy_to_sdk"], dtype=np.int32)
            if len(self.policy_to_hw) != 12:
                raise ValueError(
                    f"policy_to_sdk must have 12 entries, got {len(self.policy_to_hw)}"
                )
            print(f"[cfg] policy_to_sdk (from yaml) = {self.policy_to_hw.tolist()}")
        else:
            order_name = cfg.get("policy_joint_order", "FR_FL_RR_RL")
            self.policy_to_hw = build_policy_to_hw_mapping(order_name)
            print(f"[cfg] policy_joint_order = {order_name}")
            print(f"[cfg] policy->hw mapping = {self.policy_to_hw.tolist()}")
        print(f"[cfg] joystick cmd_max   = {self.cmd_max.tolist()}")
        print(f"[cfg] joystick cmd_sign  = {self.cmd_sign.tolist()}")

        # ---------------- policy ----------------
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_root = os.path.normpath(os.path.join(script_dir, "../LeggedGym-Ex"))
        legged_gym_root = os.environ.get("LEGGED_GYM_ROOT_DIR", default_root)
        print(f"[cfg] LEGGED_GYM_ROOT_DIR = {legged_gym_root}")

        policy_path = cfg["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", legged_gym_root)
        print(f"[cfg] loading policy: {policy_path}")
        if not os.path.isfile(policy_path):
            raise FileNotFoundError(f"Policy file not found: {policy_path}")
        self.policy = torch.jit.load(policy_path)
        self.policy.eval()

        # ---------------- runtime buffers ----------------
        self.action         = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos = self.default_angles.copy()
        self.obs            = np.zeros(self.num_obs,     dtype=np.float32)
        self.counter        = 0

        # ---------------- DDS IO ----------------
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

        # ---------------- release sport_mode ----------------
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

    # ---------- io helpers ----------

    def send_cmd(self):
        self.low_cmd.crc = self._crc.Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)

    def wait_for_low_state(self, timeout: float = 5.0):
        t0 = time.time()
        while True:
            with self._state_lock:
                ok = self._got_first_state
            if ok:
                break
            if time.time() - t0 > timeout:
                raise RuntimeError(
                    "No LowState received. Check network interface and robot."
                )
            time.sleep(self.control_dt)
        with self._state_lock:
            tick = self.low_state.tick
        print(f"[io] connected to robot, LowState flowing. (first tick={tick})")

    # ---------- state readout ----------

    def read_joint_state_policy_order(self):
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
            quat = np.array(self.low_state.imu_state.quaternion, dtype=np.float32)
            gyro = np.array(self.low_state.imu_state.gyroscope,  dtype=np.float32)
        return quat, gyro

    # ---------- cmd readout from PS5 pad ----------

    def update_cmd_from_joystick(self):
        """Map PS5 sticks -> [vx, vy, yaw_rate] with deadzone, sign flips, caps.

        Intuitive mapping (matches Unitree official docs):
            left stick forward  -> robot forward           (vx > 0)
            left stick right    -> robot strafes right     (vy < 0 by convention)
            right stick right   -> robot yaws right / CW   (yaw_rate < 0 by convention)

        Pygame convention reminder: stick "up" gives NEGATIVE axis Y,
        so vx = -ly gets you "push forward = go forward".
        """
        self.joy.poll()

        # raw -> intuitive velocity frame
        vx_raw       = -self.joy.ly             # push stick forward = +vx
        vy_raw       = -self.joy.lx             # push stick left    = +vy
        yaw_rate_raw = -self.joy.rx             # push stick left    = +yaw (CCW)

        raw = np.array([vx_raw, vy_raw, yaw_rate_raw], dtype=np.float32)
        # per-axis sign flip (if your training convention disagrees)
        raw *= self.cmd_sign
        # clamp to safe training range
        self.cmd = np.clip(raw * self.cmd_max, -self.cmd_max, self.cmd_max)

    # ---------- command write in POLICY order ----------

    def write_pd_cmd_policy_order(self, q_target, kp, kd):
        for p in range(12):
            h = int(self.policy_to_hw[p])
            m = self.low_cmd.motor_cmd[h]
            m.mode = GO2_MODE_SERVO
            m.q    = float(q_target[p])
            m.dq   = 0.0
            m.kp   = float(kp[p])
            m.kd   = float(kd[p])
            m.tau  = 0.0

    # ---------- debug reporting ----------

    LEG_NAMES = ["FL", "FR", "RL", "RR"]
    JOINT_SUFFIX = ["hip", "thigh", "calf"]

    def _print_debug_report(self, q_policy, dq_policy, gravity, gyro):
        """Rich per-step diagnostic. Compares target vs actual for every joint.

        Note on what the columns mean:
          - target      : self.target_dof_pos  (what we ASKED the motor to reach)
          - actual      : q_policy             (what the motor actually IS at)
          - err         : target - actual      (signed position error, rad)
          - dq          : dq_policy            (joint velocity, rad/s)
          - default     : self.default_angles  (ideal standing pose)
          - dev_default : actual - default     (how far policy drifts from stance)

        If the robot is standing and the policy is doing nothing crazy, dev_default
        should stay small (under ~0.2 rad on each joint while cmd=0).
        """
        target = self.target_dof_pos
        actual = q_policy
        err    = target - actual
        dev_d  = actual - self.default_angles

        # ---- per-leg summary (3 joints each) ----
        print()
        print(f"[debug] step={self.counter:5d}  "
              f"cmd=[{self.cmd[0]:+.2f},{self.cmd[1]:+.2f},{self.cmd[2]:+.2f}]  "
              f"gravity=[{gravity[0]:+.2f},{gravity[1]:+.2f},{gravity[2]:+.2f}]  "
              f"omega=[{gyro[0]:+.2f},{gyro[1]:+.2f},{gyro[2]:+.2f}]")
        print(f"         leg | joint  | target | actual | err    | dq     | default | dev")
        print(f"         ----+--------+--------+--------+--------+--------+---------+-------")
        for leg_idx in range(4):
            for j in range(3):
                p = leg_idx * 3 + j
                name = f"{self.LEG_NAMES[leg_idx]}_{self.JOINT_SUFFIX[j]:<5}"
                warn = ""
                if abs(err[p]) > 0.3:
                    warn = "  <-- large err"
                elif abs(dev_d[p]) > 0.5:
                    warn = "  <-- far from default"
                print(f"         {self.LEG_NAMES[leg_idx]:<3} | {name:<6} | "
                      f"{target[p]:+6.2f} | {actual[p]:+6.2f} | "
                      f"{err[p]:+6.2f} | {dq_policy[p]:+6.2f} | "
                      f"{self.default_angles[p]:+6.2f}  | {dev_d[p]:+6.2f}{warn}")

        # ---- aggregate health indicators ----
        max_err      = float(np.max(np.abs(err)))
        max_dev      = float(np.max(np.abs(dev_d)))
        mean_err     = float(np.mean(np.abs(err)))
        tilt_deg     = float(np.degrees(np.arctan2(
                            np.sqrt(gravity[0]**2 + gravity[1]**2),
                            abs(gravity[2]))))
        print(f"         summary: max|err|={max_err:.3f}  mean|err|={mean_err:.3f}  "
              f"max|dev_default|={max_dev:.3f}  body_tilt={tilt_deg:.1f}deg")
        if tilt_deg > 25.0:
            print(f"         🚨 body_tilt > 25deg — robot is falling / badly tilted!")

    # ---------- phases ----------


    def phase_zero_torque_until_options(self):
        print("\n[phase 1] zero-torque. Robot should be limp.")
        print("          Press  TRIANGLE (△)  on the PS5 pad to move to default stance.")
        print("          (Ctrl+C in terminal to abort.)")
        while True:
            self.joy.poll()
            if self.joy.button[PS5Map.BTN_START_STAND]:
                print("[phase 1] TRIANGLE pressed.")
                # debounce
                while self.joy.button[PS5Map.BTN_START_STAND]:
                    self.joy.poll()
                    time.sleep(0.02)
                break
            _fill_zero_torque(self.low_cmd)
            self.send_cmd()
            time.sleep(self.control_dt)

    def phase_move_to_default(self, total_time: float = 2.0):
        print(f"\n[phase 2] moving to default stance over {total_time:.1f}s ...")
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

    def phase_hold_default_until_cross(self):
        print("\n[phase 3] holding default stance.")
        print("          Press  CROSS (X)  on the PS5 pad to start the policy.")
        print("          (Ctrl+C in terminal to abort.)")
        while True:
            self.joy.poll()
            if self.joy.button[PS5Map.BTN_START_RUN]:
                print("[phase 3] CROSS pressed.")
                while self.joy.button[PS5Map.BTN_START_RUN]:
                    self.joy.poll()
                    time.sleep(0.02)
                break
            self.write_pd_cmd_policy_order(self.default_angles, self.kps, self.kds)
            self.send_cmd()
            time.sleep(self.control_dt)

    def phase_run_policy(self, debug_print: bool = True):
        print("\n[phase 4] running policy.")
        print("          Left stick  : vx (forward/back) + vy (strafe)")
        print("          Right stick : yaw_rate (turn)")
        print("          Press SQUARE to exit to damping, or Ctrl+C.")
        debug_step = max(1, int(0.5 / self.control_dt))

        try:
            while True:
                t_start = time.time()
                self.counter += 1

                # --- read joystick -> self.cmd ---
                self.update_cmd_from_joystick()

                # --- exit on SQUARE button ---
                if self.joy.button[PS5Map.BTN_EXIT]:
                    print("[run] SQUARE pressed -> exiting to damping.")
                    break

                # --- read robot state in policy order ---
                q_policy, dq_policy = self.read_joint_state_policy_order()
                quat, gyro = self.read_imu()

                # --- build observation ---
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

                # --- inference ---
                with torch.no_grad():
                    obs_t = torch.from_numpy(self.obs).unsqueeze(0)
                    act_t = self.policy(obs_t)
                self.action = act_t.detach().numpy().squeeze().astype(np.float32)
                self.action = np.clip(self.action, -100.0, 100.0)

                # --- PD target and send ---
                self.target_dof_pos = self.action * self.action_scale + self.default_angles
                self.write_pd_cmd_policy_order(self.target_dof_pos, self.kps, self.kds)
                self.send_cmd()

                # --- debug ---
                if debug_print and self.counter % debug_step == 1:
                    self._print_debug_report(q_policy, dq_policy, gravity, gyro)

                dt_left = self.control_dt - (time.time() - t_start)
                if dt_left > 0:
                    time.sleep(dt_left)
        except KeyboardInterrupt:
            print("\n[run] KeyboardInterrupt -> entering damping state.")

    def phase_damping(self, duration: float = 1.0):
        print("[phase 5] damping state.")
        n = max(1, int(duration / self.control_dt))
        for _ in range(n):
            _fill_damping(self.low_cmd, kd=3.0)
            self.send_cmd()
            time.sleep(self.control_dt)
        print("[phase 5] done. Exit.")


# ================================================================
# Joystick probe utility (use this if button/axis numbers look wrong)
# ================================================================

def probe_joystick():
    """Print live axis and button values. Ctrl+C to exit."""
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("No joystick found.")
        return
    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"Opened: {js.get_name()}")
    print(f"  axes   : {js.get_numaxes()}")
    print(f"  buttons: {js.get_numbuttons()}")
    print("Press any button / move any stick. Ctrl+C to exit.")
    try:
        while True:
            pygame.event.pump()
            axes = [round(js.get_axis(i), 2) for i in range(js.get_numaxes())]
            btns = [js.get_button(i) for i in range(js.get_numbuttons())]
            pressed = [i for i, v in enumerate(btns) if v]
            print(f"  axes={axes}  pressed_buttons={pressed}", end="\r")
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nprobe exit.")


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, nargs="?", default=None,
                        help="network interface. 'lo' for unitree_mujoco, "
                             "'eth0'/'enp...' for real robot.")
    parser.add_argument("config", type=str, nargs="?", default=None,
                        help="yaml file name under ./configs/, e.g. go2_real.yaml")
    parser.add_argument("--domain", type=int, default=0)
    parser.add_argument("--no-debug", action="store_true")
    parser.add_argument("--probe-joystick", action="store_true",
                        help="Just print joystick axes/buttons and exit.")
    args = parser.parse_args()

    if args.probe_joystick:
        probe_joystick()
        return

    if args.net is None or args.config is None:
        parser.error("net and config are required (unless --probe-joystick).")

    # joystick first: fail fast if it's missing
    joy = PS5Joystick(deadzone=0.08)

    # resolve config path
    candidates = [
        os.path.join(os.path.dirname(__file__), "configs", args.config),
        args.config,
    ]
    config_path = next((p for p in candidates if os.path.isfile(p)), None)
    if config_path is None:
        print(f"[fatal] config not found. Tried: {candidates}")
        sys.exit(1)
    print(f"[cfg] loading {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        print(f"[fatal] {config_path} is empty or contains only comments.")
        sys.exit(1)
    if not isinstance(cfg, dict):
        print(f"[fatal] {config_path} does not parse to a dict (got {type(cfg).__name__}).")
        sys.exit(1)
    required_keys = ["simulation_dt", "control_decimation",
                     "kps", "kds", "default_angles", "policy_path"]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        print(f"[fatal] {config_path} is missing required keys: {missing}")
        sys.exit(1)

    print(f"[dds] ChannelFactoryInitialize on '{args.net}' with domain_id={args.domain}")
    ChannelFactoryInitialize(args.domain, args.net)

    dep = Go2Deployer(cfg, joystick=joy)
    dep.wait_for_low_state()

    try:
        dep.phase_zero_torque_until_options()
        dep.phase_move_to_default(total_time=2.0)
        dep.phase_hold_default_until_cross()
        dep.phase_run_policy(debug_print=not args.no_debug)
    except KeyboardInterrupt:
        print("\n[main] interrupted before policy start.")
    finally:
        dep.phase_damping(duration=1.0)
        pygame.quit()


if __name__ == "__main__":
    main()