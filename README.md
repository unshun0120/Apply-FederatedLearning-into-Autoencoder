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
![image](https://github.com/user-attachments/assets/ef989b0b-f14e-4cbe-bef3-6a3437e49f62)  
  
---  
## Run  
### Train
+ using CPU to train the model  
```  
python train.py
```  
+ using GPU to train the model  
```  
python train.py --gpu=0  
```  
### Test
+ using CPU to test the model  
```  
python test.py
```  
+ using GPU to test the model  
```  
python test.py --gpu=0  
``` 
