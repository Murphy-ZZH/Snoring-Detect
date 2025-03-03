import numpy as np
import tensorflow as tf
import os
import soundfile
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

#=========超参数监控说明===========================
import wandb
wandb.init(project='TFL', entity='ljhahaha', mode='offline')
wandb.config.lr = 0.0001
wandb.config.decay = 0.0005
wandb.config.hidden_layer1 = 512
wandb.config.hidden_layer2 = 512
wandb.config.hidden_layer3 = 512
wandb.config.dropout1 = 0.5
wandb.config.dropout2 = 0.5
wandb.config.dropout3 = 0.5
wandb.config.batch_size = 32
wandb.config.epochs = 30

#=========数据读取和处理===========================
def read_audio(path, target_fs):
    (audio, fs) = soundfile.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs

#=======mel谱提取===========================
def melspec(path, target_fs, savepath):
    filelist = os.listdir(path)
    for elem in filelist:
        audio, fs = read_audio(os.path.join(path, elem), target_fs)
        melspec = librosa.feature.melspectrogram(y=audio / 32768, sr=fs, n_fft=1024, hop_length=512, n_mels=128, power=2.0)
        logmelspec = librosa.power_to_db(melspec)
        plt.figure()
        librosa.display.specshow(logmelspec, sr=fs, x_axis='time', y_axis='hz')
        plt.set_cmap('rainbow')
        plt.savefig(os.path.join(savepath, f"{os.path.splitext(elem)[0]}.png"))
        plt.close()
    return None

#=========图片预处理和标签制作====================
def melspec_processing(melspecpath):
    filelist = os.listdir(melspecpath)
    x = []
    y = []
    for file in filelist:
        filepath = os.path.join(melspecpath, file)
        print(f"Processing file: {filepath}")
        img = Image.open(filepath).convert('RGB')
        img = img.resize((224, 224), Image.LANCZOS)
        img = np.array(img)
        x.append(img)
        label = int(file.split('_')[0])  # 通过文件名解析标签
        y.append(label)
    x = np.array(x)
    y = np.array(y)
    print(f"Unique labels in y: {np.unique(y)}")  # 输出标签的种类
    return x, y

#=========迁移学习特征提取========================
def transfer_feature(x):
    m = x.shape[0]
    Resnet = tf.keras.applications.ResNet50(include_top=False)
    feature = Resnet.predict(x)
    x_out = feature.reshape(m, -1)
    return x_out

#=========构建自定义后半段模型====================
def my_model(num_class):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=wandb.config.hidden_layer1, activation=tf.keras.activations.relu),
        tf.keras.layers.Dropout(rate=wandb.config.dropout1),
        tf.keras.layers.Dense(units=wandb.config.hidden_layer2, activation=tf.keras.activations.relu),
        tf.keras.layers.Dropout(rate=wandb.config.dropout2),
        tf.keras.layers.Dense(units=wandb.config.hidden_layer3, activation=tf.keras.activations.relu),
        tf.keras.layers.Dropout(rate=wandb.config.dropout3),
        tf.keras.layers.Dense(units=num_class, activation=tf.keras.activations.sigmoid)
    ])
    return model

#=========训练模型===============================
def train(num_class, x, y, chkpath):
    x_out = transfer_feature(x)
    model = my_model(num_class)
    y = to_categorical(y, num_classes=num_class)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=wandb.config.lr, decay=wandb.config.decay),
                  loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(x_out, y, epochs=wandb.config.epochs, batch_size=wandb.config.batch_size, validation_split=0.25)
    
    # 绘制训练过程中的准确率和损失曲线
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel("Epochs")
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(chkpath + 'accury.png')
    plt.close()

    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel("Epochs")
    plt.ylabel('Value of loss function')
    plt.legend()
    plt.savefig(chkpath + 'loss.png')
    plt.close()

    wandb.log({
        "Train Accuracy": acc,
        "Train Loss": loss,
        "Test Accuracy": val_acc,
        "Test Loss": val_loss})
    
    model.save(os.path.join(wandb.run.dir, "my_model.h5"))
    return model

#=========评估和预测===========================
def predict(modelpath, modelname, x):
    model = load_model(modelpath + modelname)
    y_pred = model.predict(x)
    return y_pred

#=========正式训练============================
print('--------------Transfer learning begin------------------------------')

train_wave = "train_wave"
test_wave = "test_wave"
train_mel_save = "train_mel_save\\"
test_mel_save = "test_mel_save\\"
chkpath = "chkpath"
target_fs = 16000
num_class = 2  # 两个标签，0 和 1

# 提取梅尔谱特征
print('-------Extract melspectrogram feature of train and test set---------')
melspec(train_wave, target_fs, train_mel_save)
melspec(test_wave, target_fs, test_mel_save)

# 图片预处理和标签制作
print('-----Preprocessing of melspectrogram of training and testing-----')
x_train, y_train = melspec_processing(train_mel_save)
x_test, y_test = melspec_processing(test_mel_save)

# 数据洗牌
print('------Dataset shuffle of training--------------')
index = np.random.permutation(len(y_train))  # 打乱数据顺序
x_train = x_train[index]
y_train = y_train[index]

# 模型训练
print('-------model train and predict------------------------')
model = train(num_class, x_train, y_train, chkpath)

# 特征提取并进行预测
x_test_out = transfer_feature(x_test)
y_prob = model.predict(x_test_out)

# 评估指标计算
print('----------evaluate------------------------------')
y_test = np.array(y_test)
y_pred = np.argmax(y_prob, axis=1)

# 计算并打印评估指标
auc = roc_auc_score(y_test, y_pred) if len(np.unique(y_test)) > 1 else 0.0
print('AUC=%0.4f' % auc)

f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print('F1 score=%0.4f' % f1)
print('Precision=%0.4f' % precision)
print('Recall=%0.4f' % recall)

# 记录到 wandb
wandb.log({
    "roc_auc_score": auc,
    "F1 score": f1,
    "Precision": precision,
    "Recall": recall
})

wandb.log({
        "roc_auc_score": auc,
        "F1 score": f1,
        "Precision": precision,
        "Recall": recall
})
wandb.log({"pr": wandb.plot.pr_curve(y_test, y_prob)})
wandb.log({"roc": wandb.plot.roc_curve(y_test, y_prob)})

pass
