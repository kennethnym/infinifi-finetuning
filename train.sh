dora -P audiocraft run solver=musicgen/musicgen_base_32khz model/lm/model_scale=small continue_from=//pretrained/facebook/musicgen-small conditioner=text2music dset=audio/lofi dataset.num_workers=2 dataset.batch_size=2 optim.epochs=30 optim.lr=1e-5 schedule.cosine.warmup=8 optim.optimizer=adamw optim.adam.weight_decay=0.01