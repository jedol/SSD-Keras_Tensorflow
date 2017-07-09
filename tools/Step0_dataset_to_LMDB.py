from _init_path import *
import os
import shutil

from sample.batch_manager import binary_encoder
from tfd.LMDB import Writer
from utils.timer import Timer
from utils.mics import save_json


def create_data_dir(path, force=False):
    if os.path.isdir(path):
        if not force:
            cmd = raw_input('Function delete exist data folder. Proceed? (y/n)\n >> ')
            assert cmd == 'y', 'Job canceled...'
        shutil.rmtree(path)
    os.makedirs(path)


def dataset_to_LMDB(lmdb_path, dataset):
    ## parameters
    num_sample = len(dataset)

    ## open LMDB writer
    print 'Open LMDB writer...'
    writer = Writer(lmdb_path)

    ## write dataset to LMDB
    print 'Write dataset to LMDB...'
    sw = Timer(0, num_sample)
    for data in dataset:
        sw.tic()

        dump = binary_encoder(data)
        writer.write(dump)

        sw.toc()

    writer.close()
    print 'Job finished...'
    return num_sample


#*************************************************************************************#
## select one dataset to dump to LMDB
# dataset_name = 'VOC07'
dataset_name = 'VOC07+12'

## path to VOCdevkit
##  Note that both 'VOC2007' and 'VOC2012' folders should be in the 'VOCdevkit' folder
voc_path = '/mnt/Data/datasets/VOCdevkit'
#*************************************************************************************#

cfg = dict()
cfg['dataset'] = {'name': dataset_name}

## load DB
print 'Read annotations...'
if dataset_name == 'VOC07':
    from dataset.PASCAL_VOC import trainval_2007, test_2007, ind_to_class
    train_dataset = trainval_2007(voc_path, include_difficult=True)
    valid_dataset = test_2007(voc_path, include_difficult=True)
    i_to_c = ind_to_class()
elif dataset_name == 'VOC07+12':
    from dataset.PASCAL_VOC import trainval_2007, trainval_2012, test_2007, ind_to_class
    train_dataset = (trainval_2007(voc_path, include_difficult=True)+
                     trainval_2012(voc_path, include_difficult=True))
    valid_dataset = test_2007(voc_path, include_difficult=True)
    i_to_c = ind_to_class()
else:
    raise ValueError('Unknown dataset name: {}'.format(dataset_name))

## index(label) to class name mapper
if 'i_to_c' in locals():
    cfg['dataset']['ind_to_class'] = i_to_c

## create DB folder
force = True
lmdb_path = os.path.join(DATA_PATH, cfg['dataset']['name'])
create_data_dir(lmdb_path, force)

## dump train dataset to LMDB
cfg['dataset']['source_train'] = 'train'
train_lmdb_path = os.path.join(lmdb_path, cfg['dataset']['source_train'])
cfg['dataset']['num_train'] = dataset_to_LMDB(train_lmdb_path, train_dataset)

## dump validation dataset to LMDB, if required
if 'valid_dataset' in locals():
    cfg['dataset']['source_valid'] = 'valid'
    valid_lmdb_path = os.path.join(lmdb_path, cfg['dataset']['source_valid'])
    cfg['dataset']['num_valid'] = dataset_to_LMDB(valid_lmdb_path, valid_dataset)

## write cinfiguration to json file
cfg_path = os.path.join(lmdb_path, 'cfg.json')
save_json(cfg_path, cfg, indent=4)