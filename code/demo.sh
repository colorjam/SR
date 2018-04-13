python main.py --test --data_test Set5 --pre_train ../experiment/2018-04-13/model/model_2_best.pt --n_resblocks 2 --save --description 'test model on Set5'
# python main.py --resume 15 --epochs 5 --crop_size 128 --n_resblocks 2 --train_batch 10 --loss L1 --data_train B100 --data_test B100 --print_freq 2 --description 'continue train from epoch 2'
# python main.py --epochs 10 --crop_size 128 --n_resblocks 2 --train_batch 10 --loss Perceptual --data_train B100 --data_test B100 --print_freq 2 --description 'train with perceptual loss' --reset
