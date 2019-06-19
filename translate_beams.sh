export CUDA_VISIBLE_DEVICES=$1
MODEL_CKPT=$2
MODEL_TYPE=${3:-shallow}
srcLan=${4:-en}
tgtLan=${5:-de}

CODE_PATH=./deepNMT
TRANS_FOLDER=./${MODEL_TYPE}_trans
# specify the data path
DATA_PATH='./data/wmt14_en_de_joined_dict'   

mkdir -p $TRANS_FOLDER

beamsize=5
lenpen=1.0
output_file=$TRANS_FOLDER/trans_allbeams"$beamsize"

if [ -f $output_file ]; then
  echo $output_file "exists"
  continue
fi

python ${CODE_PATH}/generate.py \
--source-lang $srcLan \
--target-lang $tgtLan \
${DATA_PATH} \
--path ${MODEL_CKPT} \
--beam $beamsize --lenpen $lenpen --batch-size 128 --nbest $beamsize | tee $output_file

grep ^H $output_file | cut -f3- > "$output_file".sys
grep ^T $output_file | cut -f2- > "$output_file".ref
grep ^S $output_file | cut -f2- > "$output_file".src

