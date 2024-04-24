import argparse
import inspect
from loguru import logger as log
from torch.utils.data import DataLoader

from src.config import TrainConfig
from src.utils.init_method import init_dirs, init_logs, init_seeds
from src.utils.log_config import log_config
from src.data import get_dataset, get_transforms


def get_config_class(config_name):
    members = inspect.getmembers(TrainConfig, inspect.isclass)
    for name, member in members:
        if name == config_name:
            return member
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='conf')
    args = parser.parse_args()
    # 创建配置对象
    config_name = args.config
    config_class = get_config_class(config_name)
    if config_class is None:
        log.error("Config class not found. Please check the config name.")
    config = config_class()

    # 不要颠倒顺序
    sample_dir, logs_dir, tensorboard_dir = init_dirs(config)
    init_logs(config)
    init_seeds(config)
    log.info("Config: \n{}", log_config(config_class))
    dataset_train, dataset_test = get_dataset(config, transform=get_transforms(config))

    train_dataloader = DataLoader(dataset_train, batch_size=config.train_loader_batchSize, shuffle=True,
                                  num_workers=config.train_loader_num_workers)

    if dataset_test is not None:
        test_dataloader = DataLoader(dataset_test, batch_size=config.test_loader_batchSize, shuffle=True,
                                     num_workers=config.test_loader_num_workers)

