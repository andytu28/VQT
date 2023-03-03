#!/bin/bash


GPUIDX=$1
DATA_NAME=$2
NUM_CLASSES=$3
Q_PROMPT_LEN=$4
OPTIMIZER=$5
FEATURE=$6
KEEP_FRAC=$7

MODEL_ROOT=pre-trained_weights
DATA_PATH=vtab_data
OUTPUT_DIR=h2t_vit_experiments/VQTSup_${Q_PROMPT_LEN}_${OPTIMIZER}
LRP_COEF=0.0001


echo 'Compressing the model ...'
CUDA_VISIBLE_DEVICES=$GPUIDX python head2toe_sparsity_train.py \
  --train-type "h2t-prompt" \
  --config-file configs/h2t-prompt/vtab.yaml \
  --h2t_sparse_mode compress \
  MODEL.TYPE "h2t-vit" \
  MODEL.TRANSFER_TYPE "h2t-prompt" \
  DATA.BATCH_SIZE "128" \
  MODEL.H2T.NUM_QUERY_TOKENS "$Q_PROMPT_LEN" \
  MODEL.H2T.DROPOUT "0.1" \
  DATA.FEATURE $FEATURE \
  DATA.NAME $DATA_NAME \
  DATA.NUMBER_CLASSES $NUM_CLASSES \
  DATA.DATAPATH $DATA_PATH \
  MODEL.MODEL_ROOT $MODEL_ROOT \
  OUTPUT_DIR $OUTPUT_DIR \
  SOLVER.OPTIMIZER $OPTIMIZER \
  MODEL.H2T.LRP_COEF $LRP_COEF \
  MODEL.H2T.KEEP_FRAC 1.0


echo 'Feature selection and training with '$KEEP_FRAC' ...'
CUDA_VISIBLE_DEVICES=$GPUIDX python head2toe_sparsity_train.py \
  --train-type "h2t-prompt" \
  --config-file configs/h2t-prompt/vtab.yaml \
  --h2t_sparse_mode featselect \
  MODEL.TYPE "h2t-vit" \
  MODEL.TRANSFER_TYPE "h2t-prompt" \
  DATA.BATCH_SIZE "128" \
  MODEL.H2T.NUM_QUERY_TOKENS "$Q_PROMPT_LEN" \
  MODEL.H2T.DROPOUT "0.1" \
  DATA.FEATURE $FEATURE \
  DATA.NAME $DATA_NAME \
  DATA.NUMBER_CLASSES $NUM_CLASSES \
  DATA.DATAPATH $DATA_PATH \
  MODEL.MODEL_ROOT $MODEL_ROOT \
  OUTPUT_DIR $OUTPUT_DIR \
  SOLVER.OPTIMIZER $OPTIMIZER \
  MODEL.H2T.LRP_COEF 0.0 \
  MODEL.H2T.KEEP_FRAC $KEEP_FRAC
