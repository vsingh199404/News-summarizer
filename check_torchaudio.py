import torch

print('torch', torch.__version__)
print('cuda available', torch.cuda.is_available())
try:
    import torchaudio
    print('torchaudio', torchaudio.__version__)
except Exception as e:
    import traceback
    traceback.print_exc()
