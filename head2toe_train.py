import glob
import random
from random import randint
from time import sleep
import os
import torch
import numpy as np
import src.utils.logging as logging
from src.utils.file_io import PathManager
from src.configs.config import get_cfg
from src.models.build_h2t_model import build_head2toe_model
from launch import default_argument_parser, logging_train_setup
from tune_vtab import get_loaders
from src.engine.evaluator import Evaluator
from src.engine.trainer import Trainer


def get_lrwd_range(args):

    if args.train_type == "h2t-prompt":
        if args.optimizer == 'sgd':
            lr_range = [
                5.0, 2.5, 1.0,
                50.0, 25., 10.0,
                0.5, 0.25, 0.1, 0.05
            ]
            wd_range = [0.01, 0.001, 0.0001, 0.0]
        else:   # For adam basically
            if args.arch_feats == 'sup_vitl16_imagenet21k':
                lr_range = [
                    0.5, 0.1, 0.05
                ]
                wd_range = [0.01, 0.0001, 0.0]
            elif args.arch_feats == 'sup_vith14_imagenet21k':
                lr_range = [
                    0.5, 0.1, 0.05
                ]
                wd_range = [0.01, 0.0001, 0.0]
            else:
                lr_range = [
                    1.0, 0.5, 0.25, 0.1, 0.05
                ]
                wd_range = [0.01, 0.001, 0.0001, 0.0]
    else:
        raise ValueError()

    return lr_range, wd_range


def find_best_lrwd(files, data_name):
    t_name = "val_" + data_name
    best_lr = None
    best_wd = None
    best_val_acc = -1
    for f in files:
        try:
            results_dict = torch.load(f, "cpu")
            epoch = len(results_dict) - 1
            val_result = results_dict[f"epoch_{epoch}"]["classification"][t_name]["top1"]
            val_result = float(val_result)
        except Exception as e:
            print(f"Encounter issue: {e} for file {f}")
            continue

        if val_result == best_val_acc:
            frag_txt = f.split("/run")[0]
            cur_lr = float(frag_txt.split("/lr")[-1].split("_wd")[0])
            cur_wd = float(frag_txt.split("_wd")[-1].split('_bs')[0])
            if best_lr is not None and cur_lr < best_lr:
                # get the smallest lr to break tie for stability
                best_lr = cur_lr
                best_wd = cur_wd
                best_val_acc = val_result

        elif val_result > best_val_acc:
            best_val_acc = val_result
            frag_txt = f.split("/run")[0]
            best_lr = float(frag_txt.split("/lr")[-1].split("_wd")[0])
            best_wd = float(frag_txt.split("_wd")[-1].split('_bs')[0])
    return best_lr, best_wd


def setup(args, lr, wd, final_runs, run_idx=None, seed=100):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.SOLVER.DBG_TRAINABLE = True
    assert(args.optimizer == cfg.SOLVER.OPTIMIZER)

    cfg.SEED = seed

    if not final_runs:
        cfg.RUN_N_TIMES = 1
        cfg.MODEL.SAVE_CKPT = False
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + '_val'
        lr = lr / 256 * cfg.DATA.BATCH_SIZE
        cfg.SOLVER.BASE_LR = lr
        cfg.SOLVER.WEIGHT_DECAY = wd
    else:
        cfg.RUN_N_TIMES = 5  # No use. Just individually try out 5 seeds
        cfg.MODEL.SAVE_CKPT = True
        files = glob.glob(
            f'{cfg.OUTPUT_DIR}_val/{cfg.DATA.NAME}/{cfg.DATA.FEATURE}/*/'
            + 'run1/eval_results.pth'
        )
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + '_final'
        lr, wd = find_best_lrwd(files, cfg.DATA.NAME)
        cfg.SOLVER.BASE_LR = lr
        cfg.SOLVER.WEIGHT_DECAY = wd

    # Setup the output dir
    output_dir = cfg.OUTPUT_DIR
    bs = cfg.DATA.BATCH_SIZE
    output_folder = os.path.join(
            cfg.DATA.NAME, cfg.DATA.FEATURE, f'lr{lr}_wd{wd}_bs{bs}'
    )

    # Train cfg.RUN_N_TIMES times
    if run_idx is None:
        count = 1
        while count <= cfg.RUN_N_TIMES:
            output_path = os.path.join(
                    output_dir, output_folder, f'run{count}')
            sleep(randint(1, 5))
            if not PathManager.exists(output_path):
                PathManager.mkdirs(output_path)
                cfg.OUTPUT_DIR = output_path
                break
            else:
                count += 1
        if count > cfg.RUN_N_TIMES:
            raise ValueError(
                    f'Already run {cfg.RUN_N_TIMES} times for {output_folder}.'
            )

    else:
        output_path = os.path.join(
                output_dir, output_folder, f'run{run_idx}')
        if not PathManager.exists(output_path):
            PathManager.mkdirs(output_path)
            cfg.OUTPUT_DIR = output_path
        else:
            raise ValueError(
                    f'Already run run-{run_idx} for {output_dir}.'
            )

    cfg.freeze()
    return cfg


def train(cfg, args, final_runs):

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if cfg.SEED is not None:
        torch.manual_seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        random.seed(0)

    # Setup training env including loggers
    logging_train_setup(args, cfg)
    logger = logging.get_logger('visual_prompt')

    # Setup data loaders
    train_loader, val_loader, test_loader = get_loaders(
            cfg, logger, final_runs)
    logger.info('Constructing models ...')
    model, cur_device = build_head2toe_model(cfg)

    # Setup the evaluator
    logger.info('Setting up Evaluator ...')
    evaluator = Evaluator()
    logger.info("Setting up Trainer...")
    trainer = Trainer(cfg, model, evaluator, cur_device)

    if train_loader:
        trainer.train_classifier(train_loader, val_loader, test_loader)
        torch.save(
                evaluator.results,
                os.path.join(cfg.OUTPUT_DIR, 'eval_results.pth')
        )
    else:
        print('No train loader presented. Exit.')


def main(args):

    # Tuning lr and wd on the validation set
    if not args.dont_search:
        lr_range, wd_range = get_lrwd_range(args)
        for lr in sorted(lr_range, reverse=True):
            for wd in sorted(wd_range, reverse=True):
                print(f'val ==> lr {lr}, wd {wd}')
                try:
                    cfg = setup(args, lr, wd, final_runs=False)
                except ValueError:
                    continue  # Already run
                train(cfg, args, final_runs=False)

    # Final run 5 times with different seeds
    random_seeds = [42, 44, 82, 100, 800]
    for run_idx, seed in enumerate(random_seeds):
        try:
            cfg = setup(
                    args, 0.1, 0.1, final_runs=True,
                    run_idx=run_idx+1, seed=seed)
        except ValueError:
            continue  # Already run
        train(cfg, args, final_runs=True)


if __name__ == '__main__':
    parser = default_argument_parser()
    parser.add_argument('--dont_search', default=False, action='store_true',
                        help='')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='')
    parser.add_argument('--arch_feats', type=str, default='',
                        help='')
    args = parser.parse_args()
    main(args)
