import sys
import torch

def main():
    print("Python version:", sys.version)
    print("Python executable:", sys.executable)
    print("\nPyTorch Information:")
    print("Version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("Device count:", torch.cuda.device_count())
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(0))

if __name__ == "__main__":
    main()
