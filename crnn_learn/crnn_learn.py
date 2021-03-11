import os
import sys
import json
import argparse
import create_lmdb_dataset, train, test
from sklearn.model_selection import train_test_split
from utility import general_utils as utils
from utility import multi_process


_this_folder_ = os.path.dirname(os.path.abspath(__file__))
_this_basename_ = os.path.splitext(os.path.basename(__file__))[0]


def main_generate(ini, logger=None):
    utils.folder_exists(ini['gt_path'], create_=True)

    ann_fnames = sorted(utils.get_filenames(ini['ann_path'], extensions=utils.META_EXTENSION))
    logger.info(" [GENERATE] # Total file number to be processed: {:d}.".format(len(ann_fnames)))

    gt_list = []
    for idx, ann_fname in enumerate(ann_fnames):
        logger.info(" [GENERATE-OCR] # Processing {} ({:d}/{:d})".format(ann_fname, (idx+1), len(ann_fnames)))

        # Load json
        _, ann_core_name, _ = utils.split_fname(ann_fname)
        ann_core_name = ann_core_name.replace('.jpg', '')
        with open(ann_fname) as json_file:
            json_data = json.load(json_file)
            objects = json_data['objects']
            # pprint.pprint(objects)

        texts = []
        for obj in objects:
            class_name = obj['classTitle']
            if class_name != 'textline':
                continue

            text = obj['description']
            texts.append(text)

        for t_idx, text in enumerate(texts):
            gt_list.append("".join([ann_core_name + '_crop_' + '{0:03d}'.format(t_idx) + '.jpg', '\t', text]))

    with open(os.path.join(ini['gt_path'], "labels.txt"), "w", encoding="utf8") as f:
        for i in range(len(gt_list)):
            gt = gt_list[i]
            f.write("{}\n".format(gt))

    logger.info(" # {} in {} mode finished.".format(_this_basename_, OP_MODE))
    return True

def main_split(ini, logger=None):
    utils.folder_exists(ini['train_gt_path'], create_=True)
    utils.folder_exists(ini['test_gt_path'], create_=True)

    if utils.file_exists(ini['train_gt_path']):
        print(" @ Warning: train gt file path, {}, already exists".format(ini["train_gt_path"]))
        ans = input(" % Proceed (y/n) ? ")
        if ans.lower() != 'y':
            sys.exit()
    if utils.file_exists(ini['test_gt_path']):
        print(" @ Warning: test gt file path, {}, already exists".format(ini["test_gt_path"]))
        ans = input(" % Proceed (y/n) ? ")
        if ans.lower() != 'y':
            sys.exit()

    # read gt. file
    with open(os.path.join(ini['gt_path'], "labels.txt"), "r", encoding="utf8") as f:
        crnn_gt_list = f.readlines()

    # train_ratio = float(ini['train_ratio'])
    # test_ratio = (1 - train_ratio)
    # train_gt_list, test_gt_list = train_test_split(gt_list, train_size=train_ratio, random_state=2000)

    # Match CRAFT TRAIN & TEST
    craft_train_list = sorted(utils.get_filenames(ini['craft_train_path'], extensions=utils.TEXT_EXTENSIONS))
    craft_test_list = sorted(utils.get_filenames(ini['craft_test_path'], extensions=utils.TEXT_EXTENSIONS))

    crnn_train_list = []
    crnn_test_list = []
    for crnn_gt in crnn_gt_list:
        crnn_fname = crnn_gt.split('\t')[0][:-13] + '.txt'
        craft_train_fname = os.path.join(ini['craft_train_path'], 'gt_' + crnn_fname)
        craft_test_fname = os.path.join(ini['craft_test_path'], 'gt_' + crnn_fname)

        if craft_train_fname in craft_train_list:
            crnn_train_list.append(crnn_gt)
        elif craft_test_fname in craft_test_list:
            crnn_test_list.append(crnn_gt)

    # Save train.txt file
    train_fpath = os.path.join(ini['train_gt_path'], 'labels.txt')
    with open(train_fpath, 'w') as f:
        f.write(''.join(crnn_train_list))

    test_fpath = os.path.join(ini['test_gt_path'], 'labels.txt')
    with open(test_fpath, 'w') as f:
        f.write(''.join(crnn_test_list))

    logger.info(" [SPLIT] # Train : Test ratio -> {} % : {} %".format(int(len(crnn_train_list)/len(crnn_gt_list)*100), int(len(crnn_test_list)/len(crnn_gt_list)*100)))
    logger.info(" [SPLIT] # Train : Test size  -> {} : {}".format(len(crnn_train_list), len(crnn_test_list)))
    return True

def main_crop(ini, model_dir=None, logger=None):
    craft_train_list = sorted(utils.get_filenames(ini['craft_train_path'], extensions=utils.TEXT_EXTENSIONS))
    craft_test_list = sorted(utils.get_filenames(ini['craft_test_path'], extensions=utils.TEXT_EXTENSIONS))
    logger.info(" [CRAFT-TRAIN GT] # Total gt number to be processed: {:d}.".format(len(craft_train_list)))

    for craft_list in [craft_train_list, craft_test_list]:
        if craft_list is craft_train_list:
            tar_mode = 'TRAIN'
        elif craft_list is craft_test_list:
            tar_mode = 'TEST'

        available_cpus = len(os.sched_getaffinity(0))
        mp_inputs = [(craft_fpath, ini, tar_mode) for file_idx, craft_fpath in enumerate(craft_list)]

        # Multiprocess func.
        multi_process.run(func=load_craft_gt_and_save_crop_images, data=mp_inputs,
                          n_workers=available_cpus, n_tasks=len(craft_list), max_queue_size=len(craft_list))

    return True

def load_craft_gt_and_save_crop_images(craft_fpath, ini, tar_mode, print_=False):
    # load craft gt. file
    with open(craft_fpath, "r", encoding="utf8") as f:
        craft_infos = f.readlines()
        for tl_idx, craft_info in enumerate(craft_infos):
            box = craft_info.split(',')[:8]
            box = [int(pos) for pos in box]
            x1, y1, x3, y3 = box[0], box[1], box[4], box[5]

            _, core_name, _ = utils.split_fname(craft_fpath)
            img_fname = core_name.replace('gt_', '')

            if tar_mode == 'TRAIN':
                raw_img_path = os.path.join(ini['train_img_path'], img_fname + '.jpg')
                rst_fpath = os.path.join(ini['train_crop_path'],
                                         img_fname + '_crop_' + '{0:03d}'.format(tl_idx) + '.jpg')
            elif tar_mode == 'TEST':
                raw_img_path = os.path.join(ini['test_img_path'], img_fname + '.jpg')
                rst_fpath = os.path.join(ini['test_crop_path'],
                                         img_fname + '_crop_' + '{0:03d}'.format(tl_idx) + '.jpg')

            if not (utils.file_exists(raw_img_path, print_=True)):
                print("  # Raw image doesn't exists at {}".format(raw_img_path))
                continue

            img = utils.imread(raw_img_path, color_fmt='RGB')
            crop_img = img[y1:y3, x1:x3]

            if utils.file_exists(rst_fpath):
                print("  # Save image already exists at {}".format(rst_fpath))
                pass
            else:
                utils.imwrite(crop_img, rst_fpath)
                print("  #  ({:d}/{:d}) Saved at {} ".format(tl_idx, len(craft_infos), rst_fpath))

    return True

def main_create(ini, model_dir=None, logger=None):
    for tar_mode in ['TRAIN', 'TEST']:
        if tar_mode == 'TRAIN':
            crop_img_path = os.path.join(ini['train_gt_path'], 'crop_img')
            gt_fpath = os.path.join(ini['train_gt_path'], 'labels.txt')
            lmdb_path = os.path.join(ini['train_lmdb_path'])
        elif tar_mode == 'TEST':
            crop_img_path = os.path.join(ini['test_gt_path'], 'crop_img')
            gt_fpath = os.path.join(ini['test_gt_path'], 'labels.txt')
            lmdb_path = os.path.join(ini['test_lmdb_path'])

        logger.info(" [CREATE-{}] # Create lmdb dataset".format(tar_mode))
        create_lmdb_dataset.createDataset(inputPath=crop_img_path, gtFile=gt_fpath, outputPath=lmdb_path)

    return True

def main_train(ini, model_dir=None, logger=None):
    cuda_ids = ini['cuda_ids'].split(',')
    train_args = ['--train_data', ini['train_lmdb_path'],
                  '--valid_data', ini['test_lmdb_path'],
                  '--cuda', ini['cuda'],
                  '--cuda_ids', cuda_ids,
                  '--workers', ini['workers'],
                  '--batch_size', ini['batch_size'],
                  '--num_iter', ini['num_iter'],
                  # '--saved_model', ini['saved_model'],
                  '--select_data', ini['select_data'],
                  '--Transformation', ini['Transformation'],
                  '--FeatureExtraction', ini['FeatureExtraction'],
                  '--SequenceModeling', ini['SequenceModeling'],
                  '--Prediction', ini['Prediction'],
                  '--data_filtering_off',
                  # '--batch_ratio', ini['batch_ratio'],
                  '--batch_max_length', ini['batch_max_length'],
                  '--imgH', ini['imgH'],
                  '--imgW', ini['imgW'],
                  '--PAD',
                  '--hidden_size', ini['hidden_size']]

    train.main(train_args)

    return True

def main_test(ini, model_dir=None, logger=None):

    return True


def main(args):
    ini = utils.get_ini_parameters(args.ini_fname)
    logger = utils.setup_logger_with_ini(ini['LOGGER'],
                                         logging_=args.logging_, console_=args.console_logging_)

    if args.op_mode == 'GENERATE_GT':
        main_generate(ini['GENERATE_GT'], logger=logger)
    elif args.op_mode == 'SPLIT_GT':
        main_split(ini['SPLIT_GT'], logger=logger)
    elif args.op_mode == 'CROP_IMG':
        main_crop(ini['CROP_IMG'], logger=logger)
    elif args.op_mode == 'CREATE_LMDB':
        main_create(ini['CREATE_LMDB'], logger=logger)
    elif args.op_mode == 'TRAIN':
        main_train(ini['TRAIN'], model_dir=args.model_dir, logger=logger)
    elif args.op_mode == 'TEST':
        main_test(ini['TEST'], model_dir=args.model_dir, logger=logger)
    elif args.op_mode == 'TRAIN_TEST':
        ret, model_dir = main_train(ini['TRAIN'], model_dir=args.model_dir, logger=logger)
        main_test(ini['TEST'], model_dir, logger=logger)
        print(" # Trained model directory is {}".format(model_dir))
    else:
        print(" @ Error: op_mode, {}, is incorrect.".format(args.op_mode))

    return True

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--op_mode", required=True, choices=['GENERATE_GT', 'SPLIT_GT', 'CROP_IMG', 'CREATE_LMDB', 'TRAIN', 'TEST', 'TRAIN_TEST'], help="operation mode")
    parser.add_argument("--ini_fname", required=True, help="System code ini filename")
    parser.add_argument("--model_dir", default="", help="Model directory")

    parser.add_argument("--logging_", default=False, action='store_true', help="Activate logging")
    parser.add_argument("--console_logging_", default=False, action='store_true', help="Activate logging")

    args = parser.parse_args(argv)

    return args


SELF_TEST_ = True
OP_MODE = 'CROP_IMG' # GENERATE_GT / SPLIT_GT / CROP_IMG / CREATE_LMDB / TRAIN / TEST / TRAIN_TEST
INI_FNAME = _this_basename_ + ".ini"


if __name__ == "__main__":
    if len(sys.argv) == 1:
        if SELF_TEST_:
            sys.argv.extend(["--op_mode", OP_MODE])
            sys.argv.extend(["--ini_fname", INI_FNAME])
            sys.argv.extend(["--logging_"])
            sys.argv.extend(["--console_logging_"])
        else:
            sys.argv.extend(["--help"])

    main(parse_arguments(sys.argv[1:]))

