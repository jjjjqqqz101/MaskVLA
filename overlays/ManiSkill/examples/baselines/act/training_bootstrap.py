"""Apply the measured CUDA allocator budget before starting the ACT entry."""
import os
import torch

fraction = float(os.environ.get('ACT_TRAIN_MEMORY_FRACTION', '.74'))
if not 0 < fraction <= 1:
    raise ValueError('Invalid ACT_TRAIN_MEMORY_FRACTION')
torch.cuda.set_per_process_memory_fraction(fraction, device=0)
from train_triple_rgb import main
raise SystemExit(main())
