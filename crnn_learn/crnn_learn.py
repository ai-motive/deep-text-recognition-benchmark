import argparse
import json
import os
import subprocess
import sys
import datetime
import shutil
import create_lmdb_dataset
import train
from enum import Enum
from python_utils.common import general as cg, logger as cl, string as cs
from python_utils.image import general as ig, coordinates as ic
from python_utils.json import general as jg
from python_utils.multi_process import multi_process as mp


_this_folder_ = os.path.dirname(os.path.abspath(__file__))
_this_basename_ = os.path.splitext(os.path.basename(__file__))[0]

KO, TEXTLINE = 'KO', 'TEXTLINE'  # DATASET_TYPE
PREPROCESS_ALL, GENERATE_GT, SPLIT_GT, CROP_IMG, CREATE_LMDB, MERGE, TRAIN, TEST, TRAIN_TEST = \
    'PREPROCESS_ALL', 'GENERATE_GT', 'SPLIT_GT', 'CROP_IMG', 'CREATE_LMDB', 'MERGE', 'TRAIN', 'TEST', 'TRAIN_TEST'


# MODEL NAMES (craft / yolov5)
class ModelName(Enum):
    CRAFT = 0
    YOLOv5 = 1


# OBJECT NUMBERS (graph, table, ko, math)
class ObjNum(Enum):
    GRAPH         = 0
    TABLE         = 1
    KO            = 2
    MATH          = 3


def main_generate(ini, common_info, logger=None):
    # Init. local variables
    vars = {}
    for key, val in ini.items():
        vars[key] = cs.replace_string_from_dict(val, common_info)

    cg.folder_exists(vars['gt_path'], create_=True)

    ann_fnames = sorted(cg.get_filenames(vars['ann_path'], extensions=jg.META_EXTENSION))
    logger.info(" [GENERATE] # Total file number to be processed: {:d}.".format(len(ann_fnames)))

    gt_list = []
    for idx, ann_fname in enumerate(ann_fnames):
        logger.info(" [GENERATE-OCR] # Processing {} ({:d}/{:d})".format(ann_fname, (idx+1), len(ann_fnames)))

        # Load json
        _, ann_core_name, _ = cg.split_fname(ann_fname)
        ann_core_name = ann_core_name.replace('.jpg', '')
        with open(ann_fname) as json_file:
            json_data = json.load(json_file)
            objects = json_data['objects']
            # pprint.pprint(objects)

        texts = []
        for obj in objects:
            class_name = obj['classTitle']
            if class_name != common_info['tgt_class']:
                continue

            text = obj['description']
            texts.append(text.strip())

        for t_idx, text in enumerate(texts):
            gt_list.append("".join([ann_core_name + '_crop_' + '{0:03d}'.format(t_idx) + '.jpg', '\t', text]))

    with open(os.path.join(vars['gt_path'], "labels.txt"), "w", encoding="utf8") as f:
        for i in range(len(gt_list)):
            gt = gt_list[i]
            f.write("{}\n".format(gt))

    logger.info(" # {} in {} mode finished.".format(_this_basename_, OP_MODE))
    return True


def main_split(ini, common_info, logger=None):
    # Init. local variables
    vars = {}
    for key, val in ini.items():
        vars[key] = cs.replace_string_from_dict(val, common_info)

    cg.folder_exists(vars['train_gt_path'], create_=True)
    cg.folder_exists(vars['test_gt_path'], create_=True)

    if cg.file_exists(vars['train_gt_path']):
        print(" @ Warning: train gt file path, {}, already exists".format(vars["train_gt_path"]))
        # ans = input(" % Proceed (y/n) ? ")
        # if ans.lower() != 'y':
        #     sys.exit()
    if cg.file_exists(vars['test_gt_path']):
        print(" @ Warning: test gt file path, {}, already exists".format(vars["test_gt_path"]))
        # ans = input(" % Proceed (y/n) ? ")
        # if ans.lower() != 'y':
        #     sys.exit()

    # read gt. file
    with open(os.path.join(vars['gt_path'], "labels.txt"), "r", encoding="utf8") as f:
        crnn_gt_list = f.readlines()

    # train_ratio = float(ini['train_ratio'])
    # test_ratio = (1 - train_ratio)
    # train_gt_list, test_gt_list = train_test_split(gt_list, train_size=train_ratio, random_state=2000)

    # Match CRAFT TRAIN & TEST
    ref_train_list = sorted(cg.get_filenames(vars['ref_train_path'], extensions=cg.TEXT_EXTENSIONS))
    ref_test_list = sorted(cg.get_filenames(vars['ref_test_path'], extensions=cg.TEXT_EXTENSIONS))

    crnn_train_list = []
    crnn_test_list = []
    for crnn_gt in crnn_gt_list:
        crnn_fname = crnn_gt.split('\t')[0][:-13] + '.txt'
        if 'craft' in common_info['ref_dir_name']:
            ref_train_fname = os.path.join(vars['ref_train_path'], 'gt_' + crnn_fname)
            ref_test_fname = os.path.join(vars['ref_test_path'], 'gt_' + crnn_fname)
        else:
            ref_train_fname = os.path.join(vars['ref_train_path'], crnn_fname)
            ref_test_fname = os.path.join(vars['ref_test_path'], crnn_fname)

        if ref_train_fname in ref_train_list:
            crnn_train_list.append(crnn_gt)
        elif ref_test_fname in ref_test_list:
            crnn_test_list.append(crnn_gt)

    # Save train.txt file
    train_fpath = os.path.join(vars['train_gt_path'], 'labels.txt')
    with open(train_fpath, 'w') as f:
        f.write(''.join(crnn_train_list))

    test_fpath = os.path.join(vars['test_gt_path'], 'labels.txt')
    with open(test_fpath, 'w') as f:
        f.write(''.join(crnn_test_list))

    logger.info(" [SPLIT] # Train : Test ratio -> {} % : {} %".format(int(len(crnn_train_list)/len(crnn_gt_list)*100), int(len(crnn_test_list)/len(crnn_gt_list)*100)))
    logger.info(" [SPLIT] # Train : Test size  -> {} : {}".format(len(crnn_train_list), len(crnn_test_list)))
    return True


def main_crop(ini, common_info, logger=None):
    # Init. local variables
    vars = {}
    for key, val in ini.items():
        vars[key] = cs.replace_string_from_dict(val, common_info)

    model_name = common_info['ref_dir_name'].split('_')[0]

    ref_train_fpaths = sorted(cg.get_filenames(vars['ref_train_path'], extensions=cg.TEXT_EXTENSIONS))
    ref_test_fpaths = sorted(cg.get_filenames(vars['ref_test_path'], extensions=cg.TEXT_EXTENSIONS))
    ref_fpaths = ref_train_fpaths + ref_test_fpaths
    logger.info(" [CROP] # Total ref. gt file size : {:d}.".format(len(ref_fpaths)))

    if cg.file_exists(vars['train_crop_path']):
        logger.info(f" [CROP] # Train crop dir. is already exist, it's removed !!! : {vars['train_crop_path']}")
    if cg.file_exists(vars['test_crop_path']):
        logger.info(f" [CROP] # Test crop dir. is already exist, it's removed !!! : {vars['test_crop_path']}")

    for ref_fpaths in [ref_train_fpaths, ref_test_fpaths]:
        if ref_fpaths is ref_train_fpaths:
            tar_mode = TRAIN
        elif ref_fpaths is ref_test_fpaths:
            tar_mode = TEST

        available_cpus = len(os.sched_getaffinity(0))
        mp_inputs = [(ref_fpath, vars, tar_mode, model_name) for file_idx, ref_fpath in enumerate(ref_fpaths)]

        # Multiprocess func.
        mp.run(func=save_crop_images_by_reference_filepath, data=mp_inputs,
               n_workers=available_cpus, n_tasks=len(ref_fpaths), max_queue_size=len(ref_fpaths))

    return True

def save_crop_images_by_reference_filepath(ref_fpath, vars, tar_mode, model_name=ModelName.YOLOv5.name.lower()):
    # Load img info
    _, core_name, _ = cg.split_fname(ref_fpath)
    if model_name == ModelName.CRAFT.name.lower():
        img_fname = core_name.replace('gt_', '')
    elif model_name == ModelName.YOLOv5.name.lower():
        img_fname = core_name

    low_tar_mode = tar_mode.lower()  # train / test
    raw_img_path = os.path.join(vars[f'{low_tar_mode}_img_path'], img_fname + '.jpg')

    if not (cg.file_exists(raw_img_path, print_=True)):
        print("  # Raw image doesn't exists at {}".format(raw_img_path))
        return False

    img = ig.imread(raw_img_path, color_fmt='RGB')
    h, w, c = img.shape

    # load yolov5 gt. files
    text_boxes = []
    with open(ref_fpath, "r", encoding="utf8") as f:
        ref_infos = f.readlines()
        for idx, ref_info in enumerate(ref_infos):
            coco_data = ref_info.replace('\n', '').split(' ')
            if len(coco_data) != 5:
                continue
            class_num = int(coco_data[0])
            if class_num != ObjNum.KO.value:
                continue

            max_x_plus_min_x, max_x_minus_min_x  = float(coco_data[1]) * 2 * w, float(coco_data[3]) * w
            max_y_plus_min_y, max_y_minus_min_y = float(coco_data[2]) * 2 * h, float(coco_data[4]) * h

            double_min_x, double_max_x = (max_x_plus_min_x - max_x_minus_min_x), (max_x_plus_min_x + max_x_minus_min_x)
            double_min_y, double_max_y = (max_y_plus_min_y - max_y_minus_min_y), (max_y_plus_min_y + max_y_minus_min_y)
            min_x, max_x = int(double_min_x / 2), int(double_max_x / 2)
            min_y, max_y = int(double_min_y / 2), int(double_max_y / 2)

            text_boxes.append([[min_x, min_y], [max_x, max_y]])

        if text_boxes:
            for t_idx, t_box in enumerate(text_boxes):
                [[min_x, min_y], [max_x, max_y]] = t_box
                crop_img_fname = img_fname + '_crop_' + '{0:03d}'.format(t_idx)
                rst_fpath = os.path.join(vars[f'{low_tar_mode}_crop_path'], crop_img_fname + '.jpg')

                crop_img = img[min_y:max_y, min_x:max_x]

                ig.imwrite(crop_img, rst_fpath)
                print("  #  ({:d}/{:d}) Saved at {} ".format(idx, len(ref_infos), rst_fpath))
        else:
            print(f"  #  Reference gt is empty !!! : {ref_fpath}")

    return True

def main_create(ini, common_info, logger=None):
    # Init. local variables
    vars = {}
    for key, val in ini.items():
        vars[key] = cs.replace_string_from_dict(val, common_info)

    for tar_mode in [TRAIN, TEST]:
        if tar_mode == TRAIN:
            crop_img_path = os.path.join(vars['train_gt_path'], 'crop_img')
            gt_fpath = os.path.join(vars['train_gt_path'], 'labels.txt')
            lmdb_path = os.path.join(vars['train_lmdb_path'])
        elif tar_mode == TEST:
            crop_img_path = os.path.join(vars['test_gt_path'], 'crop_img')
            gt_fpath = os.path.join(vars['test_gt_path'], 'labels.txt')
            lmdb_path = os.path.join(vars['test_lmdb_path'])

        logger.info(" [CREATE-{}] # Create lmdb dataset".format(tar_mode))
        create_lmdb_dataset.createDataset(inputPath=crop_img_path, gtFile=gt_fpath, outputPath=lmdb_path)

    return True

def main_merge(ini, common_info, logger=None):
    global src_train_gt_path, src_test_gt_path, dst_train_gt_path, dst_test_gt_path
    # Init. local variables
    vars = {}
    for key, val in ini.items():
        vars[key] = cs.replace_string_from_dict(val, common_info)

    cg.folder_exists(vars['total_dataset_path'], create_=True)

    datasets = [dataset for dataset in os.listdir(vars['dataset_path']) if (dataset != 'total') and ('meta.json' not in dataset)]
    sort_datasets = sorted(datasets, key=lambda x: (int(x.split('_')[0])))

    base_dir_name = common_info['base_dir_name']

    # Process total files
    train_gt_text_paths = []
    test_gt_text_paths = []
    if len(sort_datasets) != 0:
        for dir_name in sort_datasets:

            src_train_path, src_test_path = os.path.join(vars['dataset_path'], dir_name, TRAIN.lower()), os.path.join(vars['dataset_path'], dir_name, TEST.lower())
            src_train_crop_img_path = os.path.join(src_train_path, f'{base_dir_name}/crop_img/')
            src_test_crop_img_path = os.path.join(src_test_path, f'{base_dir_name}/crop_img/')

            dst_train_path, dst_test_path = os.path.join(vars['total_dataset_path'], TRAIN.lower()), os.path.join(vars['total_dataset_path'], TEST.lower())
            dst_train_crop_img_path = os.path.join(dst_train_path, f'{base_dir_name}/crop_img/')
            dst_test_crop_img_path = os.path.join(dst_test_path, f'{base_dir_name}/crop_img/')

            if cg.folder_exists(dst_train_crop_img_path) and cg.folder_exists(dst_test_crop_img_path):
                logger.info(" # Already {} is exist".format(vars['total_dataset_path']))
            else:
                cg.folder_exists(dst_train_crop_img_path, create_=True), cg.folder_exists(dst_test_crop_img_path, create_=True)

            # Apply symbolic link for gt & img path
            for tar_mode in [TRAIN, TEST]:
                if tar_mode is TRAIN:
                    src_crop_img_path = src_train_crop_img_path
                    dst_crop_img_path = dst_train_crop_img_path
                elif tar_mode is TEST:
                    src_crop_img_path = src_test_crop_img_path
                    dst_crop_img_path = dst_test_crop_img_path

                # link img_path
                src_crop_imgs = sorted(cg.get_filenames(src_crop_img_path, extensions=ig.IMG_EXTENSIONS))
                dst_crop_imgs = sorted(cg.get_filenames(dst_crop_img_path, extensions=ig.IMG_EXTENSIONS))

                src_crop_fnames = [cg.split_fname(crop_img)[1] for crop_img in src_crop_imgs]
                dst_crop_fnames = [cg.split_fname(crop_img)[1] for crop_img in dst_crop_imgs]
                if any(src_fname not in dst_crop_fnames for src_fname in src_crop_fnames):
                    img_sym_cmd = 'find {} -name "*.jpg" -exec ln {} {} \;'.format(src_crop_img_path, '{}', dst_crop_img_path) # link each files
                    # img_sym_cmd = 'ln "{}"* "{}"'.format(src_crop_img_path, dst_crop_img_path)  # argument is long
                    subprocess.call(img_sym_cmd, shell=True)
                    logger.info(" # Link img files {} -> {}.".format(src_crop_img_path, dst_crop_img_path))
                else:
                    logger.info(" # Link img files already generated : {}.".format(dst_crop_img_path))

            # Add to list all label files
            for tar_mode in [TRAIN, TEST]:
                if tar_mode == TRAIN:
                    src_train_gt_path = os.path.join(src_train_path, f'{base_dir_name}', 'labels.txt')
                    train_gt_text_paths.append(src_train_gt_path)

                    dst_train_gt_path = os.path.join(dst_train_path, f'{base_dir_name}', 'labels.txt')

                elif tar_mode == TEST:
                    src_test_gt_path = os.path.join(src_test_path, f'{base_dir_name}', 'labels.txt')
                    test_gt_text_paths.append(src_test_gt_path)

                    dst_test_gt_path = os.path.join(dst_test_path, f'{base_dir_name}', 'labels.txt')

        logger.info(" # Train gt paths : {}".format(train_gt_text_paths))
        logger.info(" # Test gt paths : {}".format(test_gt_text_paths))

        # Merge all label files
        concat_train_gt_text_files_ = cg.concat_text_files(train_gt_text_paths, dst_train_gt_path)
        if concat_train_gt_text_files_:
            logger.info(" # Concat success : Train gt text files !!!")
        else:
            logger.info(" # Concat fail : Train gt text files !!!")


        concat_test_gt_text_files_ = cg.concat_text_files(test_gt_text_paths, dst_test_gt_path)
        if concat_test_gt_text_files_:
            logger.info(" # Concat success : Test gt text files !!!")
        else:
            logger.info(" # Concat fail : Test gt text files !!!")


        for tar_mode in [TRAIN, TEST]:
            logger.info(" [CREATE-{}] # Create lmdb dataset".format(tar_mode))
            if tar_mode == TRAIN:
                crop_img_path = dst_train_crop_img_path
                gt_fpath = dst_train_gt_path
                lmdb_path = vars['total_train_lmdb_path']
            elif tar_mode == TEST:
                crop_img_path = dst_test_crop_img_path
                gt_fpath = dst_test_gt_path
                lmdb_path = vars['total_test_lmdb_path']

            if cg.folder_exists(lmdb_path):
                shutil.rmtree(lmdb_path)
                create_lmdb_dataset.createDataset(inputPath=crop_img_path, gtFile=gt_fpath, outputPath=lmdb_path)
                logger.info(f" [CREATE-ALL] # Remove and Create all lmdb dataset : {lmdb_path}")
            else:
                create_lmdb_dataset.createDataset(inputPath=crop_img_path, gtFile=gt_fpath, outputPath=lmdb_path)
                logger.info(" [CREATE-ALL] # Create all lmdb dataset")

    return True

def main_train(ini, common_info, logger=None):
    # Init. local variables
    vars = {}
    for key, val in ini.items():
        vars[key] = cs.replace_string_from_dict(val, common_info)

    cuda_ids = vars['cuda_ids'].split(',')

    now = datetime.datetime.now()
    exp_name = now.strftime('%y%m%d')

    train_args = ['--train_data', vars['train_lmdb_path'],
                  '--valid_data', vars['test_lmdb_path'],
                  '--cuda', vars['cuda'],
                  '--cuda_ids', cuda_ids,
                  '--exp_name', exp_name,
                  '--workers', vars['workers'],
                  '--batch_size', vars['batch_size'],
                  '--num_iter', vars['num_iter'],
                  '--saved_model', vars['saved_model'],
                  '--select_data', vars['select_data'],
                  '--Transformation', vars['transformation'],
                  '--FeatureExtraction', vars['featureextraction'],
                  '--SequenceModeling', vars['sequencemodeling'],
                  '--Prediction', vars['prediction'],
                  '--data_filtering_off',
                  # '--batch_ratio', vars['batch_ratio'],
                  '--batch_max_length', vars['batch_max_length'],
                  '--imgH', vars['imgh'],
                  '--imgW', vars['imgw'],
                  '--PAD',
                  '--hidden_size', vars['hidden_size']]

    train.main(train_args)

    return True


def main(args):
    ini = cg.get_ini_parameters(args.ini_fname)
    common_info = {}
    for key, val in ini['COMMON'].items():
        common_info[key] = val

    logger = cl.setup_logger_with_ini(ini['LOGGER'],
                                         logging_=args.logging_, console_=args.console_logging_)

    if args.op_mode == PREPROCESS_ALL:
        # Init. local variables
        vars = {}
        for key, val in ini[PREPROCESS_ALL].items():
            vars[key] = cs.replace_string_from_dict(val, common_info)

        # Run generate & split
        tgt_dir_names = vars['tgt_dir_names'].replace(' ', '').split(',')
        for tgt_dir_name in tgt_dir_names:
            common_info['tgt_dir_name'] = tgt_dir_name
            main_generate(ini[GENERATE_GT], common_info, logger=logger)
            main_split(ini[SPLIT_GT], common_info, logger=logger)
            main_crop(ini[CROP_IMG], common_info, logger=logger)

        # Run merge
        main_merge(ini[MERGE], common_info, logger=logger)

    elif args.op_mode == GENERATE_GT:
        main_generate(ini[GENERATE_GT], common_info, logger=logger)
    elif args.op_mode == SPLIT_GT:
        main_split(ini[SPLIT_GT], common_info, logger=logger)
    elif args.op_mode == CROP_IMG:
        main_crop(ini[CROP_IMG], common_info, logger=logger)
    elif args.op_mode == CREATE_LMDB:
        main_create(ini[CREATE_LMDB], common_info, logger=logger)
    elif args.op_mode == MERGE:
        main_merge(ini[MERGE], common_info, logger=logger)
    elif args.op_mode == TRAIN:
        main_train(ini[TRAIN], common_info, logger=logger)
    else:
        print(" @ Error: op_mode, {}, is incorrect.".format(args.op_mode))

    return True

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", required=True, choices=[KO, TEXTLINE],
                        help="dataset type")
    parser.add_argument("--op_mode", required=True, choices=[PREPROCESS_ALL, GENERATE_GT, SPLIT_GT, CROP_IMG, CREATE_LMDB, MERGE, TRAIN, TEST, TRAIN_TEST], help="operation mode")
    parser.add_argument("--ini_fname", required=True, help="System code ini filename")

    parser.add_argument("--logging_", default=False, action='store_true', help="Activate logging")
    parser.add_argument("--console_logging_", default=False, action='store_true', help="Activate logging")

    args = parser.parse_args(argv)

    return args


SELF_TEST_ = True
DATASET_TYPE = KO  # KO / TEXTLINE
OP_MODE = PREPROCESS_ALL
# PREPROCESS_ALL
# (GENERATE_GT / SPLIT_GT / CROP_IMG / CREATE_LMDB or MERGE)
# TRAIN

INI_FNAME = _this_basename_ + ".ini"


if __name__ == "__main__":
    if len(sys.argv) == 1:
        if SELF_TEST_:
            sys.argv.extend(["--dataset_type", DATASET_TYPE])
            sys.argv.extend(["--op_mode", OP_MODE])
            sys.argv.extend(["--ini_fname", INI_FNAME])
            sys.argv.extend(["--logging_"])
            sys.argv.extend(["--console_logging_"])
        else:
            sys.argv.extend(["--help"])

    main(parse_arguments(sys.argv[1:]))


