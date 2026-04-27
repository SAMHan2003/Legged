"""
最小探测脚本:只订阅 rt/lowstate,打印 tick 和 IMU 四元数。
    python probe_lowstate.py lo       # 探测 unitree_mujoco
    python probe_lowstate.py eth0     # 探测实机
如果 5 秒内还没数据,说明数据源(模拟器/机器人)没连通,和 deploy 脚本无关。
"""
import sys
import time
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo

count = {"n": 0, "last_tick": -1}

def cb(msg: LowStateGo):
    count["n"] += 1
    if count["n"] % 50 == 1:  # ~ every second if data is at 500Hz
        print(f"[rx] tick={msg.tick}  "
              f"quat(w,x,y,z)=({msg.imu_state.quaternion[0]:.3f}, "
              f"{msg.imu_state.quaternion[1]:.3f}, "
              f"{msg.imu_state.quaternion[2]:.3f}, "
              f"{msg.imu_state.quaternion[3]:.3f})  "
              f"motor0.q={msg.motor_state[0].q:.3f}")

if __name__ == "__main__":
    net    = sys.argv[1] if len(sys.argv) > 1 else "lo"
    domain = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    print(f"[probe] ChannelFactoryInitialize on '{net}' domain_id={domain}")
    ChannelFactoryInitialize(domain, net)

    sub = ChannelSubscriber("rt/lowstate", LowStateGo)
    sub.Init(cb, 10)

    print("[probe] subscribed to rt/lowstate, waiting up to 30s ...")
    t0 = time.time()
    while time.time() - t0 < 30:
        time.sleep(0.1)
    print(f"[probe] done.  received {count['n']} messages in 30s.")
    if count["n"] == 0:
        print("[probe] ==> NO DATA.  "
              "Check that unitree_mujoco / real robot is running and uses the same interface.")
