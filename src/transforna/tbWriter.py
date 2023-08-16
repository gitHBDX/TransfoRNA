
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
writer = SummaryWriter(str(Path(__file__).parent.parent.absolute())+"/runs/")
