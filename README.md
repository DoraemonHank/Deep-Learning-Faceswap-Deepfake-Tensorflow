# Faceswap-Deepfake-Tensorflow
## Version 1
This is Tensorflow version refer to https://github.com/joshua-wu/deepfakes_faceswap which using Keras.
![image](https://github.com/DoraemonHank/Faceswap-Deepfake-Tensorflow/blob/main/image/output.jpg)
### Demo
trump to cage   
![image](https://github.com/DoraemonHank/Faceswap-Deepfake-Tensorflow/blob/main/image/afyx7-zdfcc.gif)
<br>
<br>
cage to trump
<br>
![image](https://github.com/DoraemonHank/Faceswap-Deepfake-Tensorflow/blob/main/image/x6l5w-gcv3n.gif)        
### Requirements:
    Tensorflow-GPU 1.15
    CUDA 10.0
    cuDNN 7.6.2
    Python 3.6.12
### How to run:
    1. Create file : checkpoint
    2. Download dataset: https://anonfiles.com/p7w3m0d5be/face-swap.zip
    3. Put cage and trump in to data file
    4. Training : python train_v1.py
    5. Download the video then swap the face : python predict_video.py


