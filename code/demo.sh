# python main.py --test --data_test Set5 --pre_train ../experiment/2018-04-06/model/model_20.pt --n_resblocks 2 --reset --crop_size 128
# python main.py --load 3 --epochs 17 --patch_size 128 --n_resblocks 2 --train_batch 10 --loss L1 --aug
python main.py --epochs 1 --crop_size 128 --n_resblocks 1 --train_batch 10 --loss L1 --reset