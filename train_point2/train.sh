# # Gulf
# python train.py ./config/ST_Gulf_FNN.yaml --epochs 50
# python train.py ./config/ST_Gulf_LSTM.yaml --epochs 50
# python train.py ./config/ST_Gulf_Earthformer.yaml --epochs 50
# python train.py ./config/ST_Gulf_UNET.yaml --epochs 50
# python train.py ./config/ST_Gulf_IMLP.yaml --epochs 50
# python train.py ./config/ST_Gulf_ATGRU.yaml --epochs 50

python train.py ./config/ST_Gulf_FNN.yaml --test
python train.py ./config/ST_Gulf_LSTM.yaml --test
python train.py ./config/ST_Gulf_Earthformer.yaml --test
python train.py ./config/ST_Gulf_UNET.yaml --test
python train.py ./config/ST_Gulf_IMLP.yaml --test
python train.py ./config/ST_Gulf_ATGRU.yaml --test

# Pacific
# python train.py ./config/ST_Gulf_FNN.yaml --epochs 50 --region Pacific
# python train.py ./config/ST_Gulf_LSTM.yaml --epochs 50 --region Pacific
# python train.py ./config/ST_Gulf_Earthformer.yaml --epochs 50 --region Pacific
# python train.py ./config/ST_Gulf_UNET.yaml --epochs 50 --region Pacific
# python train.py ./config/ST_Gulf_IMLP.yaml --epochs 50 --region Pacific
# python train.py ./config/ST_Gulf_ATGRU.yaml --epochs 50 --region Pacific

# python train.py ./config/ST_Gulf_FNN.yaml --test --region Pacific 
# python train.py ./config/ST_Gulf_LSTM.yaml --test --region Pacific
# python train.py ./config/ST_Gulf_Earthformer.yaml --test --region Pacific
# python train.py ./config/ST_Gulf_UNET.yaml --test --region Pacific
# python train.py ./config/ST_Gulf_IMLP.yaml --test --region Pacific
# python train.py ./config/ST_Gulf_ATGRU.yaml --test --region Pacific


# Indian
# python train.py ./config/ST_Gulf_FNN.yaml --epochs 50 --region Indian
# python train.py ./config/ST_Gulf_LSTM.yaml --epochs 50 --region Indian
# python train.py ./config/ST_Gulf_Earthformer.yaml --epochs 50 --region Indian
# python train.py ./config/ST_Gulf_UNET.yaml --epochs 50 --region Indian
# python train.py ./config/ST_Gulf_IMLP.yaml --epochs 50 --region Indian
# python train.py ./config/ST_Gulf_ATGRU.yaml --epochs 50 --region Indian

# python train.py ./config/ST_Gulf_FNN.yaml --test --region Indian
# python train.py ./config/ST_Gulf_LSTM.yaml --test --region Indian
# python train.py ./config/ST_Gulf_Earthformer.yaml --test --region Indian
# python train.py ./config/ST_Gulf_UNET.yaml --test --region Indian
# python train.py ./config/ST_Gulf_IMLP.yaml --test --region Indian
# python train.py ./config/ST_Gulf_ATGRU.yaml --test --region Indian