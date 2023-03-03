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
from src.engine.h2t_sparsity_trainer import H2TSparsityTrainer


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

    cfg.SEED = seed

    if not final_runs:
        raise NotImplementedError()   # Consider final_runs for now
        cfg.RUN_N_TIMES = 1
        cfg.MODEL.SAVE_CKPT = False
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + '_sparseval'
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
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + '_sparsefinal'
        lr, wd = find_best_lrwd(files, cfg.DATA.NAME)
        cfg.SOLVER.BASE_LR = lr * 0.5
        cfg.SOLVER.WEIGHT_DECAY = wd
        print(f'LR: {lr}, WD: {wd}')

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
                    output_dir, output_folder, f'run{count}')  ## TODO: for sparseval?
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
        if args.h2t_sparse_mode == 'compress':
            output_path = os.path.join(
                    output_dir, output_folder, f'compress/run{run_idx}')
        elif args.h2t_sparse_mode == 'featselect':
            output_path = os.path.join(
                    output_dir, output_folder,
                    f'keep_frac_{cfg.MODEL.H2T.KEEP_FRAC}/run{run_idx}')
        else:
            output_path = os.path.join(
                    output_dir, output_folder,
                    f'finetune_{cfg.MODEL.H2T.KEEP_FRAC}/run{run_idx}')

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

    # Auto set the load weights path
    if args.h2t_sparse_mode in ['featselect', 'compress']:
        args.load_weights = os.path.join(
                cfg.OUTPUT_DIR.replace('_sparsefinal', '_final'), 'last_model.pth')
        args.load_weights = args.load_weights.replace(
                'compress/', '').replace(
                        f'keep_frac_{cfg.MODEL.H2T.KEEP_FRAC}/', '')
    else:
        args.load_weights = os.path.join(
                cfg.OUTPUT_DIR, 'last_model.pth')
        args.load_weights = args.load_weights.replace(
                f'finetune_', f'keep_frac_')

    assert(os.path.exists(args.load_weights))

    # Load trained weights before compression
    if args.load_weights != '':
        logger.info(f'Load weights from {args.load_weights}')
        content = torch.load(args.load_weights, map_location='cpu')

        # Remove the head from the loaded weights
        if args.h2t_sparse_mode in ['featselect', 'compress']:
            content['model'].pop('head.last_layer.weight')
            content['model'].pop('head.last_layer.bias')

        missing_keys = []
        for n, p in model.state_dict().items():
            if n in content['model'] and content['model'][n].size() == p.size():
                p.data.copy_(content['model'][n])
            else:
                missing_keys.append(n)
        logger.info(
                f'Chkpt missing keys: {missing_keys} (should be the head only or none)'
        )

    logger.info("Setting up H2TSparsityTrainer...")
    trainer = H2TSparsityTrainer(
            cfg, args.h2t_sparse_mode, model, evaluator, cur_device)

    if train_loader:
        if args.h2t_sparse_mode == 'compress':
            # 1. train with MODEL.H2T.LRP_COEF > 0.0, MODEL.H2T.KEEP_FRAC = 1.0
            assert(cfg.MODEL.H2T.KEEP_FRAC == 1.0 and cfg.MODEL.H2T.LRP_COEF > 0.0)
            trainer.train_classifier(train_loader, val_loader, test_loader,
                                     lrp_coef=cfg.MODEL.H2T.LRP_COEF, feat_select_ids=None)
        elif args.h2t_sparse_mode in ['featselect', 'finetune']:
            # 2. train with MODEL.H2T.LRP_COEF = 0.0, MODEL.H2T.KEEP_FRAC < 1.0
            #  OR
            # 3. fine-tune the classifier and the query prompts
            assert(cfg.MODEL.H2T.KEEP_FRAC <= 1.0 and cfg.MODEL.H2T.LRP_COEF == 0.0)
            if args.h2t_sparse_mode == 'featselect':
                compress_path = os.path.join(cfg.OUTPUT_DIR.replace(
                        f'keep_frac_{cfg.MODEL.H2T.KEEP_FRAC}', 'compress'), 'last_model.pth')
            else:
                compress_path = os.path.join(cfg.OUTPUT_DIR.replace(
                        f'finetune_{cfg.MODEL.H2T.KEEP_FRAC}', 'compress'), 'last_model.pth')
            assert(os.path.exists(compress_path))
            content = torch.load(compress_path, map_location='cpu')
            scores = torch.norm(content['model']['head.last_layer.weight'], p=2, dim=0)
            feat_dim = model.head.last_layer.weight.shape[1]
            _, feat_select_ids = torch.topk(scores, k=feat_dim)
            feat_select_ids = feat_select_ids.long().to(cur_device)
            trainer.train_classifier(train_loader, val_loader, test_loader,
                                     lrp_coef=0.0, feat_select_ids=feat_select_ids)
        else:
            raise NotImplementedError()

        torch.save(
                evaluator.results,
                os.path.join(cfg.OUTPUT_DIR, 'eval_results.pth')
        )
    else:
        print('No train loader presented. Exit.')


def main(args):

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
    parser.add_argument('--h2t_sparse_mode', type=str, required=True,
                        choices=['compress', 'featselect', 'finetune'],
                        help='')
    parser.add_argument('--load_weights', type=str, default='',
                        help='')
    args = parser.parse_args()
    main(args)
