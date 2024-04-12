import os
import warnings
from dataclasses import asdict

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from transforna.inference_benchmark import infer_benchmark
from transforna.inference_tcga import infer_tcga
from transforna.train.train import compute_cv, train

warnings.filterwarnings("ignore")

@hydra.main(config_path="./configs",config_name="main_config")
def my_app(cfg: DictConfig) -> None:
    path = os.getcwd()

    #validate that config is configured properly
    #assert_config(cfg)

    #init train and model config
    train_cfg_path = {"_target_": "configs.train_model_configs.%s.GeneEmbeddTrainConfig"%cfg["task"]}
    model_cfg_path = {"_target_": "configs.train_model_configs.%s.GeneEmbeddModelConfig"%cfg["task"]}
    train_config = instantiate(train_cfg_path)
    model_config = instantiate(model_cfg_path)

    #prepare configs as structured dicts
    train_config = OmegaConf.structured(asdict(train_config))
    model_config = OmegaConf.structured(asdict(model_config))

    #update model config with the name of the model 
    model_config["model_input"] = cfg["model_name"]


    #append train and model config to the main config
    cfg = OmegaConf.merge({"train_config":train_config,"model_config":model_config}, cfg)
    #inference or train
    if cfg["inference"]:
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
