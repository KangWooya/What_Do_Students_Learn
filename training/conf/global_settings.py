""" configurations for this project

author baiyu
"""
import os
from datetime import datetime

#CIFAR100 dataset path (python version)
#CIFAR100_PATH = '/nfs/private/cifar100/cifar-100-python'

#mean and std of cifar100 dataset
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

#CIFAR100_TEST_MEAN = (0.5088964127604166, 0.48739301317401956, 0.44194221124387256)
#CIFAR100_TEST_STD = (0.2682515741720801, 0.2573637364478126, 0.2770957707973042)

#directory to save weights file (override with WDSL_CKPT, e.g. a NAS path on a cluster)
CHECKPOINT_PATH = os.environ.get('WDSL_CKPT', 'checkpoint')

#total training epoches (override with WDSL_EPOCH: 300 = best config, 200 = CD-200ep)
EPOCH = int(os.environ.get('WDSL_EPOCH', '300'))
#LR-decay milestones at 30% / 60% / 80% of training (→ [90,180,240] for 300ep, [60,120,160] for 200ep)
MILESTONES = [int(EPOCH * 0.3), int(EPOCH * 0.6), int(EPOCH * 0.8)]

#initial learning rate
#INIT_LR = 0.1

DATE_FORMAT = '%d_%B_%Y_%Hh'
#time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

#tensorboard log dir
LOG_DIR = 'runs'

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 200








