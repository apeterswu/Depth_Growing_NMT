ARCH=transformer_deep_vaswani_wmt_en_de_big
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

DATA_PATH=./data/wmt14_en_de_joined_dict
CODE_PATH=./deepNMT
MODEL_PATH=./model/wmt14_e2d_"$encL"L"$decL"L_deep

encL=6
decL=6

python ${CODE_PATH}/train_deep.py ${DATA_PATH} \
--arch $ARCH --share-all-embeddings \
--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
--lr 5e-04 --min-lr 1e-09 \
--dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--save-interval-updates 10000 --max-update 800000 --keep-interval-updates 100 \
--max-tokens 4096 --no-progress-bar \
--save-dir $MODEL_PATH \
--seed 2048 \
--restore-file checkpoint_best.pt \
--update-freq 2 \
--encoder-layers $encL \
--decoder-layers $decL \
--use-encoder-cross-block-resnet \
--use-decoder-cross-block-resnet \
--additional-encoder-layers 2 \
--additional-decoder-layers 2 \
--not-reload-aux
