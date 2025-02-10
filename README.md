# Apply Federated Learning into Autoencoder and its variants

this project is based on :  
*Federated Learning*  
+ https://github.com/AshwinRJ/Federated-Learning-PyTorch  
  
*Autoencoder*  
+ https://github.com/L1aoXingyu/pytorch-beginner/tree/master    

---  
## Before Running  
1. Create a new folder called **"Dataset"** outside the downloaded folder  
2. Create a new folder called **"logs"** outside the downloaded folder  
3. Create a new folder called **"save_objects"** inside the downloaded folder  
4. Create a new folder called **"save_models"** outside the downloaded folder
![image](https://github.com/user-attachments/assets/bf822c39-64ac-4307-b7e5-de1071ed8988)  
  
---  
## Run  
### Train (train.py include the test)
+ using CPU
```  
python train.py
```  
+ using GPU 
```  
python train.py --gpu=0  
```  
+ e.g. global epoch = 1, model = Convolutional Autoencoder  
```
python train.py --gpu=0 --global_ep=1 --model=cnnae  
```

### Test
+ using CPU  
```  
python test.py
```  
+ using GPU  
```  
python test.py --gpu=0  
``` 
+ e.g. test Autoencoder model
```
python test.py --gpu=0 --model=ae  
```