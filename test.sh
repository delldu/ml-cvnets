# export EVAL_DIR='' # Location of results 
# export CKPT_NAME='' # Name of the pre-trained model weight file (e.g., checkpoint_ema.pt)
# CUDA_VISIBLE_DEVICES=0 \
# 	python main_eval.py \
# 	--common.config-file config/classification/mobilevit_small.yaml \
# 	--model.classification.pretrained \
# 	mobilevit_s.pt


# pip install setuptools==59.5.0

CUDA_VISIBLE_DEVICES=0 \
	python main_eval.py \
	--common.config-file results/mobilevit_small.yaml \
	--common.results-loc results \
	--model.classification.pretrained mobilevit_s.pt
