#!/bin/sh
# 获取当前脚本文件的上两级目录路径，并切换到该路径。
# 如果切换失败（即目录不存在等情况），则会退出脚本执行
cd $(dirname $(dirname "$0")) || exit 
ROOT_DIR=$(pwd)  # 获取当前工作目录(Present Working Directory)
PYTHON=python

TRAIN_CODE=train.py

DATASET=scannet
CONFIG="None"
EXP_NAME=debug
WEIGHT="None"
RESUME=False
GPU=None


while getopts "p:d:c:n:w:g:r:" opt; do
  case $opt in
    p)
      PYTHON=$OPTARG
      ;;
    d)
      DATASET=$OPTARG
      ;;
    c)
      CONFIG=$OPTARG
      ;;
    n)
      EXP_NAME=$OPTARG
      ;;
    w)
      WEIGHT=$OPTARG
      ;;
    r)
      RESUME=$OPTARG
      ;;
    g)
      GPU=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG"
      ;;
  esac
done

if [ "${NUM_GPU}" = 'None' ]
then
  NUM_GPU=`$PYTHON -c 'import torch; print(torch.cuda.device_count())'`
fi

echo "Experiment name: $EXP_NAME"
echo "Python interpreter dir: $PYTHON"
echo "Dataset: $DATASET"
echo "Config: $CONFIG"
echo "GPU Num: $GPU"

EXP_DIR=exp/${DATASET}/${EXP_NAME}
MODEL_DIR=${EXP_DIR}/model
CODE_DIR=${EXP_DIR}/code
CONFIG_DIR=configs/${DATASET}/${CONFIG}.py


echo " =========> CREATE EXP DIR <========="
echo "Experiment dir: $ROOT_DIR/$EXP_DIR"

if ${RESUME}
then
  CONFIG_DIR=${EXP_DIR}/config.py
  WEIGHT=$MODEL_DIR/model_last.pth
else
  mkdir -p "$MODEL_DIR" "$CODE_DIR" # mkdir 命令用于创建目录。-p 选项表示如果指定的目录不存在，将会递归创建它
  cp -r scripts tools pointcept "$CODE_DIR" # 复制文件或目录 -r 表示递归
fi

echo "Loading config in:" $CONFIG_DIR
export PYTHONPATH=./$CODE_DIR
echo "Running code in: $CODE_DIR"


echo " =========> RUN TASK <========="

if [ "${WEIGHT}" = "None" ]
then
    $PYTHON "$CODE_DIR"/tools/$TRAIN_CODE \
    --config-file "$CONFIG_DIR" \
    --num-gpus "$GPU" \
    --options save_path="$EXP_DIR"
else
    $PYTHON "$CODE_DIR"/tools/$TRAIN_CODE \
    --config-file "$CONFIG_DIR" \
    --num-gpus "$GPU" \
    --options save_path="$EXP_DIR" resume="$RESUME" weight="$WEIGHT"
fi