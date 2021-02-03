import os
import sys
import json
import argparse
from sklearn.model_selection import train_test_split
from utility import general_utils as utils


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
        # if ann_core_name == img_core_name + img_ext: # 뒤에 .json
        with open(ann_fname) as json_file:
            json_data = json.load(json_file)
            objects = json_data['objects']
            # pprint.pprint(objects)

        for obj in objects:
            class_name = obj['classTitle']
            if class_name != 'textline':
                continue

            text = obj['description']
            gt_list.append("".join([ann_core_name+'.jpg', '\t', text]))

    with open(
        os.path.join(ini['gt_path'], "labels.txt"), "w", encoding="utf8"
    ) as f:
        for i in range(len(gt_list)):
            gt = gt_list[i]
            f.write("{}\n".format(gt))

    logger.info(" # {} in {} mode finished.".format(_this_basename_, OP_MODE))
    return True

def main_split(ini, logger=None):
    utils.folder_exists(ini['train_path'], create_=True)
    utils.folder_exists(ini['test_path'], create_=True)

    if utils.file_exists(ini['train_path']):
        print(" @ Warning: train gt file path, {}, already exists".format(ini["train_path"]))
        ans = input(" % Proceed (y/n) ? ")
        if ans.lower() != 'y':
            sys.exit()
    if utils.file_exists(ini['test_path']):
        print(" @ Warning: test gt file path, {}, already exists".format(ini["test_path"]))
        ans = input(" % Proceed (y/n) ? ")
        if ans.lower() != 'y':
            sys.exit()

    # read gt. file
    with open(
        os.path.join(ini['gt_path'], "labels.txt"), "r", encoding="utf8"
    ) as f:
        gt_list = f.readlines()

    train_ratio = float(ini['train_ratio'])
    test_ratio = (1 - train_ratio)
    train_gt_list, test_gt_list = train_test_split(gt_list, train_size=train_ratio, random_state=2000)

    # Save train.txt file
    train_fpath = os.path.join(ini['train_path'], 'labels.txt')
    with open(train_fpath, 'w') as f:
        f.write(''.join(train_gt_list))

    test_fpath = os.path.join(ini['test_path'], 'labels.txt')
    with open(test_fpath, 'w') as f:
        f.write(''.join(test_gt_list))

    logger.info(" [SPLIT] # Train : Test ratio) -> {} % : {} %".format(int(train_ratio*100), int(test_ratio*100)))
    logger.info(" [SPLIT] # Train : Test size  -> {} : {}".format(len(train_gt_list), len(test_gt_list)))
    return True

def main_train(ini, model_dir=None, logger=None):

    return True

def main_test(ini, model_dir=None, logger=None):

    return True


def main(args):
    ini = utils.get_ini_parameters(args.ini_fname)
    logger = utils.setup_logger_with_ini(ini['LOGGER'],
                                         logging_=args.logging_, console_=args.console_logging_)

    if args.op_mode == 'GENERATE':
        main_generate(ini['GENERATE'], logger=logger)
    elif args.op_mode == 'SPLIT':
        main_split(ini['SPLIT'], logger=logger)
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

    parser.add_argument("--op_mode", required=True, choices=['GENERATE', 'SPLIT', 'TRAIN', 'TEST', 'TRAIN_TEST'], help="operation mode")
    parser.add_argument("--ini_fname", required=True, help="System code ini filename")
    parser.add_argument("--model_dir", default="", help="Model directory")

    parser.add_argument("--logging_", default=False, action='store_true', help="Activate logging")
    parser.add_argument("--console_logging_", default=False, action='store_true', help="Activate logging")

    args = parser.parse_args(argv)

    return args


SELF_TEST_ = True
OP_MODE = 'SPLIT' # GENERATE / SPLIT / TRAIN / TEST / TRAIN_TEST
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

