import mujoco
m = mujoco.MjModel.from_xml_path("/home/sam/Legged/unitree_mujoco/unitree_robots/go2/scene.xml")
print(f"nq={m.nq}, nv={m.nv}, nu={m.nu}, njnt={m.njnt}")
print("\n=== Joints ===")
for i in range(m.njnt):
    jnt_type = ["FREE","BALL","SLIDE","HINGE"][m.jnt_type[i]]
    print(f"  [{i}] {m.joint(i).name:25s} type={jnt_type} qpos_adr={m.jnt_qposadr[i]} dof_adr={m.jnt_dofadr[i]}")
print("\n=== Actuators ===")
for i in range(m.nu):
    jnt_id = m.actuator_trnid[i, 0]
    print(f"  [{i}] {m.actuator(i).name:25s} -> joint: {m.joint(jnt_id).name}")