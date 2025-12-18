import logging
PARALLEL_DEVICES: int = 0
USE_CUDA = True



def check_and_install_torch():
    global PARALLEL_DEVICES
    try:
        import torch.cuda
        PARALLEL_DEVICES = torch.cuda.device_count()
        logging.info(f"found {PARALLEL_DEVICES} gpus")
    except Exception:
        logging.info("can't import package 'torch'")

