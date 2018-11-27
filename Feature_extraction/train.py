# -- coding: utf-8 --
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras
from keras.datasets import cifar10
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
import numpy as np


classes_num = 10
batch_size = 32
epochs_num = 200

def quality_classify_model():
    model = Sequential()
    model.add(Flatten(input_shape=(4,4,512)))# 4*4*512
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes_num, activation='softmax'))  # 多分类

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def train():
    # 数据载入
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # 多分类标签生成
    y_train = keras.utils.to_categorical(y_train, classes_num)
    y_test = keras.utils.to_categorical(y_test, classes_num)
    # 生成训练数据
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    datagan = ImageDataGenerator(rescale=1./255)

    # 加载预训练好的卷积基
    conv_base = VGG16(include_top=False, weights='imagenet')

    # 用预训练好的卷积基处理训练集提取特征
    sample_count = len(y_train)
    train_features = np.zeros(shape=(sample_count, 4, 4, 512))
    train_labels = np.zeros(shape=(sample_count, classes_num))
    train_generator = datagan.flow(x_train, y_train, batch_size=batch_size)
    i = 0
    for inputs_batch, labels_batch in train_generator:
        features_batch = conv_base.predict(inputs_batch)
        train_features[i * batch_size : (i + 1) * batch_size] = features_batch
        train_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    # train_features = np.reshape(train_features, (sample_count, 4*4*512))

    # 用预训练好的卷积基处理验证集提取特征
    sample_count = len(y_test)
    test_generator = datagan.flow(x_test, y_test, batch_size=batch_size)
    test_features = np.zeros(shape=(sample_count, 4, 4, 512))
    test_labels = np.zeros(shape=(sample_count, classes_num))
    i = 0
    for inputs_batch, labels_batch in test_generator:
        features_batch = conv_base.predict(inputs_batch)
        test_features[i * batch_size : (i + 1) * batch_size] = features_batch
        test_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    # test_features = np.reshape(test_features, (sample_count, 4*4*512))

    model = quality_classify_model()

    # hist = model.fit_generator(train_datagan.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch = 8000, epochs = epochs_num, validation_data=(x_test,y_test), shuffle=True)
    hist = model.fit(train_features, train_labels, batch_size=batch_size, epochs=epochs_num, validation_data=(test_features, test_labels))

    model.save('./extract_features/cifar10_model.hdf5') 
    model.save_weights('./extract_features/cifar10_model_weight.hdf5')

    hist_dict = hist.history
    print("train acc:")
    print(hist_dict['acc'])
    print("validation acc:")
    print(hist_dict['val_acc'])

    train_acc = hist.history['acc']
    val_acc = hist.history['val_acc']
    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    # 绘图
    epochs = range(1, len(train_acc)+1)
    plt.plot(epochs, train_acc, 'bo', label = 'Training acc')
    plt.plot(epochs, val_acc, 'r', label = 'Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig("accuracy.png")
    plt.figure() # 新建一个图
    plt.plot(epochs, train_loss, 'bo', label = 'Training loss')
    plt.plot(epochs, val_loss, 'r', label = 'Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig("loss.png")
    