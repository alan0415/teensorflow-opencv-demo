# 1. File includeing
- cpkt: saving checkpoint during traing. Only work on problem 4.4 Training for 50 epoch.
- images: image file source
- 50epoch.png: Figure saved from previous training result for 50 epochs
- LeNet5_model_50epoch.h: Model saved on previous training for 50 epochs. The model train for 50 epochs will svaed as name "LeNet5_model.h5"
- HW1_main.ui: UI file used for building Mainwindow GUI


# 2. The program files
a. main.py
Entery for the code. Including solution for Problem 1.1 to 4.4. Other problems solution call network_train.py to implement.

b. network_train.py
It is used to work out Problem 5, about Tensorflow 2.0 usage.
Here I keep some setting to help user to make sure that the compputer work fine.

When push button, "5.4 Show Training Result", the program will run to training for 50 epcohs. This action takes much time and it will also slow down other service on computer.
To make sure the training work fine, I keep process bar and message about loading chechpoint show on screen.If you like to keep them silence, just mark up the code on line 162 to 165, also mdoify the verbose option into "verbose=0"on line 174.

The push button, "5.5 Inference", will load the model saved by "5.4 Show Training Result". So you have to train a model by yourself before inference.
If you like to try it first, you can modify the code on line 72 and 74. Remove the mark(#) on line 72 and mark up line 74 will load the model provided to test inference before training it first.

# 3. Environments
Tensorflow==2.0.0
Tensorflow-datasets==1.3.0
PyQt5==5.13.0
numpy==1.17.3
Matplotlib==3.1.1
opencv-contrib-python==3.4.2.17

# 4. Quick start
python3 main.py
