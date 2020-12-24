# garbage classify

# 准备阶段

```shell script
pip install -r requirements.txt
```
# 训练

```shell script
nohup python train.py \
             --data_path train_data/ \
             --result_path result/ \
             --gpus 0 \ # multi-gpu demo: 0,1,2,3
             --batch_size 10 \
             --epoch 150 >model_train.log 2>&1 &
```
