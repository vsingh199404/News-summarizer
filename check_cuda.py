import sys
import torch

def main():
    print("Python:", sys.version.splitlines()[0])
    print("torch:", torch.__version__)
    try:
        cuda_available = torch.cuda.is_available()
    except Exception as e:
        print("Error checking CUDA availability:", e)
        cuda_available = False
    print("CUDA available:", cuda_available)
    print("torch built CUDA version:", torch.version.cuda)
    try:
        cudnn_ver = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    except Exception:
        cudnn_ver = None
    print("cuDNN version:", cudnn_ver)
    if cuda_available:
        try:
            count = torch.cuda.device_count()
            print("CUDA device count:", count)
            for i in range(count):
                print(f"Device {i}:", torch.cuda.get_device_name(i))
        except Exception as e:
            print("Error querying CUDA devices:", e)

if __name__ == '__main__':
    main()
