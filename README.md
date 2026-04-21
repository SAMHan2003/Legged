# Go2 Mujoco Deployment Guide

## Overview

This repository provides deployment scripts for running a JIT-compiled PyTorch policy trained in Legged Gym inside MuJoCo.

**Key features:**

- Loads trained `.pt` policy
- Reconstructs training-time observations
- Matches joint ordering between policy and Mujoco
- Applies PD control for sim-to-real consistency

> ⚠️ **Important:** Deployment correctness heavily depends on matching training parameters exactly.

---

## Dependencies

Install required Python packages:

```bash
pip install mujoco torch numpy pyyaml
```

---

## Required Files

### 1. Trained Policy

- Exported from:

  ```
  play.py → exported/*.pt
  ```

### 2. Mujoco Model

- File: `scene.xml`
- Source: Unitree Mujoco repository
- Must contain:
  - 12 hinge joints (4 legs × 3 DOF)
  - 12 actuators

### 3. Deployment Config

- File: `configs/go2.yaml`
- Includes:
  - Observation scales
  - Action scales
  - PD gains
  - Command settings

---

## Configuration Reference (CRITICAL)

All parameters **must match training values**.

Otherwise, you will see:

- robot shaking
- unstable gait
- immediate falling

---

## Step-by-Step Deployment

### 1 Extract Ground-Truth Training Parameters

Before first deployment, dump actual training parameters:

```bash
cd /home/sam/Legged
python3 deploy/dump_params.py --task=go2 --num_envs=1
```

This generates:

```
configs/go2_train_params.yaml
```

 Copy values into:

```
configs/go2.yaml
```

### 2 Verify Mujoco Joint Order

Use:

```bash
python3 check_joints.py
```

Ensure:

- 12 joints detected
- Correct naming (FR, FL, RR, RL)
- Consistent ordering with policy

> **Note:** The script uses `mj_name2id`, but manual verification is strongly recommended when debugging.

### 3 Motion Commands

Modify `cmd_init` in YAML:

```yaml
# forward
cmd_init: [0.3, 0.0, 0.0]

# backward
cmd_init: [-0.3, 0.0, 0.0]

# rotate in place
cmd_init: [0.0, 0.0, 1.0]

# turn while walking
cmd_init: [0.3, 0.0, 0.5]
```

### 4 Run Deployment

```bash
python3 deploy_mujoco.py go2.yaml
```

---

## Common Issues & Debug Tips

if Robot falls immediately

- Mismatch in:
  - observation scale
  - PD gains
  - action scale

**Fix:** check params

### Random kicking / unstable motion

- Joint order mismatch
- Wrong torque mapping

**Fix:**

- print joint mapping
- compare with training env

### Simulation looks different from training

- Missing observation terms
- Incorrect gravity projection
- Different timestep

---

## Notes

- Deployment is **NOT** plug-and-play
- Sim2Real gap still exists even in Mujoco
- Always verify:
  - observation construction
  - action scaling
  - joint mapping

---

## Summary

Correct deployment requires alignment across:

- ✔ Policy
- ✔ Observation
- ✔ Action
- ✔ Dynamics

> If any of these mismatch → robot will fail
