import os
import subprocess
import shutil

cfg_file = "inv_NACA0012.cfg"
aoa_list = [0, 2, 4, 6, 8, 10]

base_output = "dataset"

for aoa in aoa_list:
    print(f"Running AoA = {aoa}")

    case_dir = os.path.join(base_output, f"AoA_{aoa}")
    os.makedirs(case_dir, exist_ok=True)

    # OPTIONAL: modify AoA inside config if needed later
    # (for now assuming cfg already updates AoA externally)

    # run SU2
    subprocess.run(["SU2_CFD", cfg_file])

    # move outputs safely
    for f in ["flow.vtu", "surface_flow.csv", "restart_flow.dat"]:
        if os.path.exists(f):
            shutil.move(f, os.path.join(case_dir, f))
