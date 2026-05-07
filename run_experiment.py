import subprocess
import sys

def main():
    print("=== CFD SURROGATE TRAINING START ===")

    subprocess.run([
        sys.executable, 
        "train.py",
        "--epochs", "100",
        "--data", "data/raw/cfd_demo_data.csv"
    ], check=True)

    print("=== TRAINING COMPLETE ===")


if __name__ == "__main__":
    main()