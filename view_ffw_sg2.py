import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path("robotis_mujoco_menagerie/robotis_ffw/scene_ffw_sg2.xml")
data = mujoco.MjData(model)

print(f"FFW-SG2 loaded")
print(f"  Bodies   : {model.nbody}")
print(f"  Joints   : {model.njnt}")
print(f"  Actuators: {model.nu}")
print(f"  DOF      : {model.nv}")
print()
print("Joint names:")
for i in range(model.njnt):
    print(f"  [{i:2d}] {model.joint(i).name}")
print()
print("Controls: mouse to orbit, scroll to zoom, right-click to pan. Close window to exit.")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.002)
