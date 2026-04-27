"""
Go2 Sim2Real 部署脚本
基于 LeggedGym-Ex 训练的模型，通过 unitree_sdk2py 控制真实 Go2

使用方法:
  python3 deploy_go2.py [网卡名]
  例: python3 deploy_go2.py eth0

观测向量 (45维):
  base_lin_vel (3) | projected_gravity (3) | base_ang_vel (3) |
  commands (3) | dof_pos_diff (12) | dof_vel (12) | last_actions (12)
"""

import time
import sys
import math
import numpy as np
import torch

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient

# ─────────────────────────────────────────────
#  配置参数（与训练配置完全对齐）
# ─────────────────────────────────────────────

# 模型路径（把 policy_1.pt 放到同目录下）
POLICY_PATH = "/home/sam/Legged/LeggedGym-Ex/logs/go2/Apr12_11-24-38_simple_rl_isaacgym/exported/Apr12_11-24-38_simple_rl_isaacgym_ite-1.pt"     # 真机上就放脚本同目录即可;也支持 {LEGGED_GYM_ROOT_DIR} 占位符


# 控制频率
CONTROL_DT = 0.02          # 50 Hz，与训练 dt 一致

# PD 增益（与 common_cfgs.py 一致）
Kp = 20.0
Kd = 0.5

# 动作缩放（与训练一致）
ACTION_SCALE = 0.25

# 观测缩放（与 legged_robot_config.py 一致）
OBS_SCALES = {
    "lin_vel": 1.0,
    "ang_vel": 0.25,
    "dof_pos": 1.0,
    "dof_vel": 0.05,
}

# 速度指令 [lin_vel_x, lin_vel_y, ang_vel_yaw]（可修改）
COMMAND = [0.5, 0.0, 0.0]  # 前进 0.5 m/s

# 观测裁剪
CLIP_OBS = 100.0

# 默认关节角度（与 common_cfgs.py init_state 一致）
# 顺序严格按照训练时 dof_names：
# FR_hip, FR_thigh, FR_calf,
# FL_hip, FL_thigh, FL_calf,
# RR_hip, RR_thigh, RR_calf,
# RL_hip, RL_thigh, RL_calf
DEFAULT_DOF_POS = np.array([
    0.0,  0.8, -1.5,   # FL
    0.0,  0.8, -1.5,   # FR
    0.0,  0.8, -1.5,   # RL
    0.0,  0.8, -1.5,   # RR
], dtype=np.float32)

# SDK 电机索引顺序（unitree_sdk2py 的 motor_state/motor_cmd 固定顺序）
# 0:FL_hip 1:FL_thigh 2:FL_calf
# 3:FR_hip 4:FR_thigh 5:FR_calf
# 6:RL_hip 7:RL_thigh 8:RL_calf
# 9:RR_hip 10:RR_thigh 11:RR_calf

# 训练 dof_names 顺序 → SDK 电机索引的映射
# 训练顺序: FR(3,4,5) FL(0,1,2) RR(9,10,11) RL(6,7,8)
POLICY_TO_SDK = [0, 1, 2,    # FL
                 3, 4, 5,    # FR
                 6, 7, 8,    # RL
                 9, 10, 11]  # RR

# SDK → 训练顺序的逆映射
SDK_TO_POLICY = [0] * 12
for policy_idx, sdk_idx in enumerate(POLICY_TO_SDK):
    SDK_TO_POLICY[sdk_idx] = policy_idx

# 站立过渡时间（秒）
STAND_DURATION = 2.0

PosStopF = math.pow(10, 9)
VelStopF = 16000.0


class Go2Deploy:
    def __init__(self, policy_path: str):
        # 加载模型
        print(f"[INFO] 加载模型: {policy_path}")
        self.policy = torch.jit.load(policy_path, map_location="cpu")
        self.policy.eval()
        print("[INFO] 模型加载成功")

        # 状态
        self.low_state: LowState_ = None
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.crc = CRC()

        self.last_actions = np.zeros(12, dtype=np.float32)
        self.start_dof_pos = np.zeros(12, dtype=np.float32)

        self.phase = "wait"       # wait → stand → policy
        self.phase_start_time = 0.0
        self.first_run = True
        self.motiontime = 0

        self.control_thread = None

    def init(self):
        """初始化通信"""
        self._init_low_cmd()

        # 发布者
        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()

        # 订阅者
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self._state_callback, 10)

        # 切换到 lowlevel 模式
        self.sc = SportClient()
        self.sc.SetTimeout(5.0)
        self.sc.Init()

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        print("[INFO] 等待传感器数据...")
        while self.low_state is None:
            time.sleep(0.1)

        print("[INFO] 释放高层控制模式...")
        status, result = self.msc.CheckMode()
        while result['name']:
            self.sc.StandDown()
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)

        print("[INFO] 已切换到 lowlevel 模式")

    def start(self):
        """启动控制线程"""
        self.phase = "stand"
        self.phase_start_time = time.time()

        # 记录起始关节角
        for sdk_idx in range(12):
            self.start_dof_pos[sdk_idx] = self.low_state.motor_state[sdk_idx].q

        self.control_thread = RecurrentThread(
            interval=CONTROL_DT,
            target=self._control_step,
            name="deploy_ctrl"
        )
        self.control_thread.Start()
        print("[INFO] 控制线程启动，开始站立过渡...")

    # ──────────────────────────────────────────
    #  内部方法
    # ──────────────────────────────────────────

    def _init_low_cmd(self):
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(20):
            self.low_cmd.motor_cmd[i].mode = 0x01
            self.low_cmd.motor_cmd[i].q   = PosStopF
            self.low_cmd.motor_cmd[i].kp  = 0
            self.low_cmd.motor_cmd[i].dq  = VelStopF
            self.low_cmd.motor_cmd[i].kd  = 0
            self.low_cmd.motor_cmd[i].tau = 0

    def _state_callback(self, msg: LowState_):
        self.low_state = msg

    def _get_obs(self) -> torch.Tensor:
        """从 low_state 构造 45 维观测向量"""
        state = self.low_state

        # IMU 数据
        imu = state.imu_state
        quat = np.array([imu.quaternion[1], imu.quaternion[2],
                         imu.quaternion[3], imu.quaternion[0]])  # xyzw
        ang_vel = np.array(imu.gyroscope, dtype=np.float32)     # 机体系角速度
        
        # 从四元数计算 projected_gravity（重力在机体系的投影）
        projected_gravity = self._quat_rotate_inverse(
            quat, np.array([0.0, 0.0, -1.0])
        ).astype(np.float32)

        # 关节状态（SDK 顺序 → 训练顺序）
        dof_pos_sdk = np.array([state.motor_state[i].q  for i in range(12)], dtype=np.float32)
        dof_vel_sdk = np.array([state.motor_state[i].dq for i in range(12)], dtype=np.float32)

        dof_pos = dof_pos_sdk[POLICY_TO_SDK]   # 转换为训练顺序
        dof_vel = dof_vel_sdk[POLICY_TO_SDK]

        # 线速度：Go2 lowlevel 没有直接提供，用零代替（训练时 lin_vel 的 scale=1.0）
        # 注意：这会带来一定误差，但 policy 有足够的鲁棒性
        base_lin_vel = np.zeros(3, dtype=np.float32)

        # 指令
        commands = np.array(COMMAND, dtype=np.float32)
        commands_scale = np.array([OBS_SCALES["lin_vel"],
                                   OBS_SCALES["lin_vel"],
                                   OBS_SCALES["ang_vel"]], dtype=np.float32)

        # 拼接观测（45维，去掉 base_lin_vel）
        obs = np.concatenate([
            projected_gravity,                                        # 3
            ang_vel * OBS_SCALES["ang_vel"],                         # 3
            commands * commands_scale,                               # 3
            (dof_pos - DEFAULT_DOF_POS) * OBS_SCALES["dof_pos"],    # 12
            dof_vel * OBS_SCALES["dof_vel"],                         # 12
            self.last_actions,                                        # 12
        ]) 

        obs = np.clip(obs, -CLIP_OBS, CLIP_OBS)
        return torch.from_numpy(obs).unsqueeze(0).float()

    def _quat_rotate_inverse(self, quat_xyzw, vec):
        """将向量从世界系转到机体系（四元数逆旋转）"""
        x, y, z, w = quat_xyzw
        # 旋转矩阵的转置（即逆旋转）
        R = np.array([
            [1-2*(y*y+z*z),   2*(x*y+w*z),   2*(x*z-w*y)],
            [  2*(x*y-w*z), 1-2*(x*x+z*z),   2*(y*z+w*x)],
            [  2*(x*z+w*y),   2*(y*z-w*x), 1-2*(x*x+y*y)]
        ])
        return R.T @ vec

    def _control_step(self):
        """每 20ms 调用一次的控制步"""
        if self.low_state is None:
            return

        now = time.time()

        # ── 阶段1：平滑过渡到站立姿态 ──
        if self.phase == "stand":
            elapsed = now - self.phase_start_time
            alpha = min(elapsed / STAND_DURATION, 1.0)

            for sdk_idx in range(12):
                policy_idx = SDK_TO_POLICY[sdk_idx]
                target = (1 - alpha) * self.start_dof_pos[sdk_idx] + \
                         alpha * DEFAULT_DOF_POS[policy_idx]
                self.low_cmd.motor_cmd[sdk_idx].q   = float(target)
                self.low_cmd.motor_cmd[sdk_idx].dq  = 0.0
                self.low_cmd.motor_cmd[sdk_idx].kp  = Kp
                self.low_cmd.motor_cmd[sdk_idx].kd  = Kd
                self.low_cmd.motor_cmd[sdk_idx].tau = 0.0

            if alpha >= 1.0:
                print("[INFO] 站立完成，切换到 Policy 控制！")
                self.phase = "policy"

        # ── 阶段2：Policy 推理控制 ──
        elif self.phase == "policy":
            obs = self._get_obs()

            with torch.no_grad():
                actions = self.policy(obs).squeeze(0).numpy()

            actions = np.clip(actions, -100.0, 100.0)
            self.last_actions = actions.copy()

            # 计算目标关节角
            target_dof_pos = DEFAULT_DOF_POS + ACTION_SCALE * actions  # 训练顺序

            # 发送指令（转换为 SDK 顺序）
            for policy_idx in range(12):
                sdk_idx = POLICY_TO_SDK[policy_idx]
                self.low_cmd.motor_cmd[sdk_idx].q   = float(target_dof_pos[policy_idx])
                self.low_cmd.motor_cmd[sdk_idx].dq  = 0.0
                self.low_cmd.motor_cmd[sdk_idx].kp  = Kp
                self.low_cmd.motor_cmd[sdk_idx].kd  = Kd
                self.low_cmd.motor_cmd[sdk_idx].tau = 0.0

        # 计算 CRC 并发送
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)


def main():
    print("=" * 50)
    print("  Go2 Sim2Real 部署脚本")
    print("=" * 50)
    print("⚠️  警告：请确保机器人周围没有障碍物！")
    print("⚠️  建议首次运行时用绳子悬空机器人！")
    print()
    input("确认安全后按 Enter 继续...")

    # 初始化通信
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    deployer = Go2Deploy(POLICY_PATH)
    deployer.init()

    print()
    print(f"[INFO] 速度指令: vx={COMMAND[0]}, vy={COMMAND[1]}, wz={COMMAND[2]}")
    print("[INFO] 2秒后开始站立...")
    time.sleep(2)

    deployer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] 收到 Ctrl+C，停止控制")
        sys.exit(0)


if __name__ == "__main__":
    main()