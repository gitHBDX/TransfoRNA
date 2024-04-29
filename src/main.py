import os
import warnings
from dataclasses import asdict

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from transforna.inference_benchmark import infer_benchmark
from transforna.inference_tcga import infer_tcga
from transforna.train.train import compute_cv, train
from hydra.core.hydra_config import HydraConfig
import sys
import logging

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# define handler and formatter
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
# add formatter to handler
handler.setFormatter(formatter)
# add handler to logger
logger.addHandler(handler)

def add_config_to_sys_path():
    cfg = HydraConfig.get()
    config_path = [path["path"] for path in cfg.runtime.config_sources if path["schema"] == "file"][0]
    sys.path.append(config_path)


@hydra.main(config_path='../conf', config_name="main_config")
def my_app(cfg: DictConfig) -> None:
    add_config_to_sys_path()

    path = os.getcwd()
    #init train and model config
    cfg['train_config'] = instantiate(cfg['train_config']).__dict__
    cfg['model_config'] = instantiate(cfg['model_config']).__dict__

    #update model config with the name of the model 
    cfg['model_config']["model_input"] = cfg["model_name"]

    #inference or train
    if cfg["inference"]:
        logger.info(f"Started inference on {cfg['task']}")
        if cfg['task'] == 'tcga':
            return infer_tcga(cfg,path=path)
        else:
            return infer_benchmark(cfg,path=path)
    else:
        if cfg["cross_val"]:
            compute_cv(cfg,path)

        else:
            train(cfg,path=path)
    
if __name__ == "__main__":
    my_app()
