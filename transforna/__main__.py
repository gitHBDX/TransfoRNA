import logging
import os
import sys
import warnings

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig

from transforna import compute_cv, infer_benchmark, infer_tcga, train

warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)

def add_config_to_sys_path():
    cfg = HydraConfig.get()
    config_path = [path["path"] for path in cfg.runtime.config_sources if path["schema"] == "file"][0]
    sys.path.append(config_path)

#transforna could called from anywhere:
#python -m transforna --config-dir = /path/to/configs 
@hydra.main(config_path='../conf', config_name="main_config")
def main(cfg: DictConfig) -> None:
    add_config_to_sys_path()
    #get path of hydra outputs folder
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    path = os.getcwd()
    #init train and model config
    cfg['train_config'] = instantiate(cfg['train_config']).__dict__
    cfg['model_config'] = instantiate(cfg['model_config']).__dict__

    #update model config with the name of the model 
    cfg['model_config']["model_input"] = cfg["model_name"]

    #inference or train
    if cfg["inference"]:
        print(f"Started inference on {cfg['task']}")
        if cfg['task'] == 'tcga':
            return infer_tcga(cfg,path=path)
        else:
            return infer_benchmark(cfg,path=path)
    else:
        if cfg["cross_val"]:
            compute_cv(cfg,path,output_dir=output_dir)

        else:
            train(cfg,path=path,output_dir=output_dir)
    
if __name__ == "__main__":
    main()
