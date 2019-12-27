# RealSR

This is an unofficial implementation of RealSR(Pytorch version).

We implement the model training and testing code based on [EDSR](https://github.com/thstkdgus35/EDSR-PyTorch).

The model and training cofiguration can be found in [config.py](https://github.com/Alan-xw/RealSR/blob/master/config.py)  
## training
For training, you can use the command below to train the model from beginning.
```
python main.py --reset
```

## testing
For testing,you can use the command below to test the model on benchmarks.
```
python main.py --test-only True
```
## demo 
For demo,you can generate super-resolved images by your trained model.
```
python main.py --demo True
```
