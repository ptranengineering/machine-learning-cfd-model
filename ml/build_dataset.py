import pandas as pd
import glob

X = []
y = []

for folder in glob.glob("dataset/AoA_*"):
    aoa = float(folder.split("_")[-1])

    df = pd.read_csv(folder + "/surface_flow.csv")

    # extract final CL/CD row
    cl = df["CL"].iloc[-1]
    cd = df["CD"].iloc[-1]

    X.append([aoa])
    y.append([cl, cd])

print("Dataset size:", len(X))