import logging
from configs.load_configs import isim_config
PARALLEL_DEVICES: int = 0
#print("="*30,"deepdoc","="*30)
if isim_config["model_plan"]["DEEPDOC_DEVICE"].lower() == "cuda":
    USE_CUDA = True
    #print("deepdoc模型部署设备为cuda")
else:
    USE_CUDA = False
    #print("deepdoc模型部署设备为cpu")


def check_and_install_torch():
    global PARALLEL_DEVICES
    try:
        import torch.cuda
        PARALLEL_DEVICES = torch.cuda.device_count()
        logging.info(f"found {PARALLEL_DEVICES} gpus")
    except Exception:
        logging.info("can't import package 'torch'")

