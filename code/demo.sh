python main.py --test --data_test Demo --pre_train ../experiment/2018-04-16/model/model_1_best.pt --n_resblocks 2 --save --description 'test model on Set5' --upscale 2
# python main.py --resume 15 --epochs 5 --crop_size 128 --n_resblocks 2 --train_batch 10 --loss L1 --data_train B100 --data_test B100 --print_freq 2 --description 'continue train from epoch 2'
# python main.py --epochs 2 --crop_size 64 --n_resblocks 2 --train_batch 10 --loss L1 --data_train DIV2K --data_test B100 --print_freq 2 --description 'train & test in DIV2K' --reset --upscale 2
