ARCH=transformer_vaswani_wmt_en_de_big
export CUDA_VISIBLE_DEVICES=$1

DATA_PATH=./data/wmt14_en_de_joined_dict
CODE_PATH=./deepNMT
nvidia-smi

SRC_MODEL_CKPT=$2
SRC_MODEL_TYPE=${3:-shallow}

TGT_MODEL_CKPT=${4:-"$SRC_MODEL_CKPT"}
TGT_MODEL_TYPE=${5:-"$SRC_MODEL_TYPE"}

suffix="${SRC_MODEL_TYPE}"_"${TGT_MODEL_TYPE}".score
beamsize=5

python ${CODE_PATH}/eval_nmt.py $DATA_PATH \
--path ${TGT_MODEL_CKPT} \
--source-file ./${SRC_MODEL_TYPE}_trans/trans_allbeams"$beamsize".src \
--target-file ./${SRC_MODEL_TYPE}_trans/trans_allbeams"$beamsize".sys \
--score-file  ./${SRC_MODEL_TYPE}_trans/trans_allbeams"$beamsize"_$suffix \
--dup-src $beamsize \
--dup-tgt 1 \
--source-lang en --target-lang de \
--max-tokens 4096
