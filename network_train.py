import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import datasets


# training recored
train_loss = []
batch = []
large_train_loss = []
large_test_accuracy = []
large_train_accuracy = []

def get_trainImg(set):
    DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
    path = tf.keras.utils.get_file('mnist', DATA_URL)   
    with np.load(path) as data:
        train_examples = data['x_train']
        train_labels = data['y_train']
        test_examples = data['x_test']
        test_labels = data['y_test']

    if (set):
        return train_examples
    else:
        sample_number = np.random.randint(60000,size=10)
    
        fig, axis = plt.subplots(1, 10)
        for i in range (10):
            axis[i].imshow(train_examples[sample_number[i],:-1], cmap="gray")
            axis[i].axis('off')
            axis[i].set_title(str(train_labels[sample_number[i]]))

        plt.show()
        
def show_index(index):
    DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
    path = tf.keras.utils.get_file('mnist', DATA_URL)   
    with np.load(path) as data:
        train_examples = data['x_train']
        train_labels = data['y_train']
        test_examples = data['x_test']
        test_labels = data['y_test']
    return train_examples[index,:-1]

def create_model(model, lrate):
    if (model == "LeNet5"):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(6, 5, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.02),input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(16, 5, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.02)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(120, activation='relu'),
            tf.keras.layers.Dense(84, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        #print("Model build success!! ")
        model.compile(optimizer=tf.keras.optimizers.SGD(lrate),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        return model

def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255

        return image, label

class Predict():
    def __init__(self):
        # For demo, loading original model
        #self.md = tf.keras.models.load_model("LeNet5_model_50epoch.h5")
        # loading model just training
        self.md = tf.keras.models.load_model("LeNet5_model.h5")

    def predict(self, index):
        DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
        path = tf.keras.utils.get_file('mnist', DATA_URL)
        (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data(path=path)
        train_images = train_images.reshape((60000, 28, 28, 1))

        train_examples = train_images / 255.0
        flatten_img = np.reshape(train_examples[index], (28, 28, 1))
        x = np.array([flatten_img])
        
        y = self.md.predict(x)

        img = show_index(index)
        
        fig, axis = plt.subplots(1, 2)
        # Origin fig
        axis[0].imshow(img, cmap='gray')
        axis[0].axis('off')
        ## show label
        # axis[0].set_title(str(train_labels[index]))

        # predict result plot        
        x_index = np.array([0,1,2,3,4,5,6,7,8,9])
        axis[1].bar(x_index, y[0])  
        
        # plot fig setting
        plt.xticks(range(10))  # 設定x刻度
        plt.yticks([0,0.2,0.4,0.6,0.8,1.0])  # 設定y刻度
        
        plt.show()
        

def LeNet_5(epoch_num, plot_type):
    # Parameter setup
    BUFFER_SIZE = 10240 # Use a much larger value for real code.
    BATCH_SIZE = 100
    NUM_EPOCHS = epoch_num
    #STEPS_PER_EPOCH = 500
    lr = 0.001
    opt = tf.keras.optimizers.SGD(lr)

    # load dataset
    DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
    path = tf.keras.utils.get_file('mnist', DATA_URL)
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255
    
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')
    
    # test data load
    mnist_data, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
    mnist_train, mnist_test = mnist_data["train"], mnist_data["test"]
    train_data = mnist_train.map(scale).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    test_data = mnist_test.map(scale).batch(BATCH_SIZE)



    if (plot_type == "iteration"):
        # model create
        net = create_model("LeNet5", lr)
        #net.summary()

        # net compile
        net.compile(optimizer=tf.keras.optimizers.SGD(lr),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        metrics_names = net.metrics_names

        for epoch in range(NUM_EPOCHS):
            #Reset the metric accumulators
            net.reset_metrics()

            for image_batch, label_batch in train_data:
                global train_loss
                result = net.train_on_batch(image_batch, label_batch)
                train_loss.append(result[0])
            global batch
            batch = np.linspace(1, 60000 / BATCH_SIZE, 60000 / BATCH_SIZE)
            plt.plot(batch, train_loss)
            plt.title('epoch [0/50]')
            plt.xlabel('iteration')
            plt.ylabel('train loss')
            plt.show()

    if (plot_type == "epoch"):
        # model create
        net2 = create_model("LeNet5", lr)
        #net2.summary()

        # net compile
        net2.compile(optimizer=tf.keras.optimizers.SGD(lr),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        metrics_names = net2.metrics_names

        # checkpoint set
        def train_and_checkpoint(net, manager):
            ckpt.restore(manager.latest_checkpoint)
            if manager.latest_checkpoint:
                print("Restored from {}".format(manager.latest_checkpoint))
            else:
                print("Initializing from scratch.")

        ckpt = tf.train.Checkpoint(optimizer = opt, model=net2)
        manager = tf.train.CheckpointManager(ckpt, directory='./cpkt', max_to_keep=1)

        # loop for training 50 epoch
        for i in range(NUM_EPOCHS):
            # load ckpt
            train_and_checkpoint(net2, manager)
            
            history = net2.fit(x_train, y_train, epochs=1, batch_size = 32, verbose=1)
            large_train_loss.append(history.history['loss'][len(history.history['loss']) - 1])
            large_train_accuracy.append(history.history['accuracy'][len(history.history['accuracy']) - 1])
            test_history = net2.evaluate(x_test, y_test, verbose = 0)
            large_test_accuracy.append(test_history[1])

            manager.save()
        
        net2.save('LeNet5_model.h5')

        plt.subplot(2,1,1)
        plt.suptitle('Accuracy')
        plt.plot(np.linspace(0,epoch_num, epoch_num), large_train_accuracy, color = 'blue', label='Training')
        plt.plot(np.linspace(0,epoch_num, epoch_num), large_test_accuracy, color = 'orange', label='Testing')
        plt.legend(loc='best')
        
        plt.subplot(2,1,2)
        plt.plot(np.linspace(0,epoch_num, epoch_num), large_train_loss, color = 'blue')
        plt.xlabel('epoch')
        
        plt.show()
