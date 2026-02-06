"""
Main entry point for Crop Leaf Disease DCGAN Project
"""

import os
import sys


def show_menu():
    print("\n=== Crop Leaf Disease DCGAN ===")
    print("1. Train DCGAN")
    print("2. Generate Images (Inference)")
    print("3. Visualize Generated Samples")
    print("4. Exit")


def run_script(script_name):
    """
    Runs a Python script using the same interpreter
    """
    command = f'"{sys.executable}" "{script_name}"'
    os.system(command)


def main():
    while True:
        show_menu()
        choice = input("Enter your choice (1-4): ")

        if choice == "1":
            print("Starting DCGAN training...")
            run_script("src/train_dcgan.py")

        elif choice == "2":
            print("Running inference...")
            run_script("src/inference.py")

        elif choice == "3":
            print("Visualizing samples...")
            run_script("src/visualization.py")

        elif choice == "4":
            print("Exiting project.")
            break

        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()
