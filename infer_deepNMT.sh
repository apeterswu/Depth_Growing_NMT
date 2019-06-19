GPU=$1
SHALLOW_MODEL_CKPT=$2
DEEP_MODEL_CKPT=$3

echo "Translating shallow model"
MODEL_TYPE="shallow"
bash translate_beams.sh $GPU ${SHALLOW_MODEL_CKPT} ${MODEL_TYPE}
bash evaluate_nmt_ppl.sh $GPU ${SHALLOW_MODEL_CKPT} ${MODEL_TYPE}

echo "Translating deep model"
MODEL_TYPE="deep"
bash translate_beams.sh $GPU ${DEEP_MODEL_CKPT} ${MODEL_TYPE}
bash evaluate_nmt_ppl.sh $GPU ${DEEP_MODEL_CKPT} ${MODEL_TYPE}

echo "Evaluating the translation scores"
SRC_MODEL_TYPE="shallow"
TGT_MODEL_TYPE="deep"
bash evaluate_nmt_ppl.sh $GPU ${SHALLOW_MODEL_CKPT} ${SRC_MODEL_TYPE} ${DEEP_MODEL_CKPT} ${TGT_MODEL_TYPE}
SRC_MODEL_TYPE="deep"
TGT_MODEL_TYPE="shallow"
bash evaluate_nmt_ppl.sh $GPU ${DEEP_MODEL_CKPT} ${SRC_MODEL_TYPE} ${SHALLOW_MODEL_CKPT} ${TGT_MODEL_TYPE}

echo "Rerank and calculate the BLEU score"
python rerank_cal_bleu.py

