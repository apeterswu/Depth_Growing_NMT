SHALLOW_MODEL=$1
DEEP_FOLDER=$2
DEEP_MODEL=$3

SCRIPT=./deepNMT/fairseq/scripts/build_initial_ckpt_for_deep.py

# example for initialize deep en-de model
python $SCRIPT \
 $SHALLOW_MODEL \
 $DEEP_FOLDER/DEEP_MODEL \
 $DEEP_FOLDER/checkpoint_best.pt
