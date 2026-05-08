#!/bin/bash

# Angles of attack for the sweep.
AOA_LIST=(0 2 4 6 8 10)

# Output dataset for ML.
echo "AoA,CL,CD" > aero_dataset.csv

for aoa in "${AOA_LIST[@]}"; do
    echo "Running AoA = $aoa"

    # SU2 v8 config key is AOA, not ANGLE_OF_ATTACK.
    sed -i -E "s/^AOA=.*/AOA= ${aoa}/" inv_NACA0012.cfg

    # Run solver and keep full log per AoA.
    if ! SU2_CFD inv_NACA0012.cfg > "output_${aoa}.txt" 2>&1; then
        echo "SU2 failed at AoA=${aoa}; writing NA row."
        echo "${aoa},NA,NA" >> aero_dataset.csv
        continue
    fi

    # Parse robustly from forces breakdown instead of fragile stdout token positions.
    CL=$(awk '/^Total CL:/{print $3; exit}' forces_breakdown.dat)
    CD=$(awk '/^Total CD:/{print $3; exit}' forces_breakdown.dat)

    if [[ -z "$CL" || -z "$CD" ]]; then
        echo "Could not parse CL/CD at AoA=${aoa}; writing NA row."
        CL="NA"
        CD="NA"
    fi

    echo "${aoa},${CL},${CD}" >> aero_dataset.csv
done

echo "Sweep complete. Dataset saved to aero_dataset.csv"
