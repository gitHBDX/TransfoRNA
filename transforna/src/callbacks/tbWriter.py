
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(str(Path(__file__).parent.parent.parent.absolute())+"/runs/")
