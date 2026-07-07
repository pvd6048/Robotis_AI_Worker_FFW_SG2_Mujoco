import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path("robotis_mujoco_menagerie/robotis_ffw/scene_ffw_sg2.xml")
data = mujoco.MjData(model)

# Map actuator names to indices
actuator_names = [model.actuator(i).name for i in range(model.nu)]

def set_ctrl(name, value):
    idx = actuator_names.index(name)
    data.ctrl[idx] = value

# T-pose: arms outstretched horizontally
set_ctrl("arm_l_joint1", 0.0)
set_ctrl("arm_l_joint2", 1.57)   # left arm out to the side
set_ctrl("arm_l_joint3", 0.0)
set_ctrl("arm_l_joint4", -1.57)
set_ctrl("arm_l_joint5", 0.0)
set_ctrl("arm_l_joint6", 0.0)
set_ctrl("arm_l_joint7", 0.0)

set_ctrl("arm_r_joint1", 0.0)
set_ctrl("arm_r_joint2", -1.57)  # right arm out to the side
set_ctrl("arm_r_joint3", 0.0)
set_ctrl("arm_r_joint4", 1.57)
set_ctrl("arm_r_joint5", 0.0)
set_ctrl("arm_r_joint6", 0.0)
set_ctrl("arm_r_joint7", 0.0)

# Head and lift at neutral
set_ctrl("head_joint1", 0.0)
set_ctrl("head_joint2", 0.0)
set_ctrl("lift_joint",  -0.25)

print("Holding T-pose. Joint states (updating every second):\n")

with mujoco.viewer.launch_passive(model, data) as viewer:
    last_print = -1.0
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()

        if data.time - last_print >= 1.0:
            last_print = data.time
            print(f"t = {data.time:.2f}s")
            for name in [
                "arm_l_joint1","arm_l_joint2","arm_l_joint3","arm_l_joint4",
                "arm_l_joint5","arm_l_joint6","arm_l_joint7",
                "arm_r_joint1","arm_r_joint2","arm_r_joint3","arm_r_joint4",
                "arm_r_joint5","arm_r_joint6","arm_r_joint7",
            ]:
                jid = model.joint(name).id
                idx = model.jnt_qposadr[jid]
                pos = data.qpos[idx]
                vel = data.qvel[model.jnt_dofadr[jid]]
                print(f"  {name:20s}  pos={pos:7.4f} rad  vel={vel:7.4f} rad/s")
            print()

        time.sleep(0.002)
