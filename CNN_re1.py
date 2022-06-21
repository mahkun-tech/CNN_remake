import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import sklearn.utils as sk_utils
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
from scipy import signal
from sklearn.model_selection import train_test_split
from swan import pycwt
from tensorflow.keras import layers
from tensorflow.keras import utils as keras_utils
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from attention_module import ChannelAttention_Module, SpatialAttention_Module
from FReLU_module import FReLU

# GPU_settings
K.clear_session()
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(
            device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")


# マルチGPU
mirrored_strategy = tf.distribute.MirroredStrategy(
    cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

# FP16_setting
# FP16を使う場合はunit数などを8の倍数に設定する
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)

# global_setting
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
# AUTOTUNEはGPUの処理とCPUの処理の配分を動的に設定してくれるパラメータ
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
EPOCH = 100
n_classes = 3

# image_size
time = 90
highest = 224
width = 224
channel = 3


@tf.function
def flip_left_right(image, label):
    return tf.image.random_flip_left_right(image), label


@tf.function
def flip_up_down(image, label):
    return tf.image.random_flip_up_down(image), label


@tf.function
def rotate_tf(image, label):
    if image.shape.__len__() == 4:

        random_angles = tf.random.uniform(shape=(tf.shape(image)[0], ), minval=-30*np
                                          .pi / 180, maxval=30*np.pi / 180)
    if image.shape.__len__() == 3:
        random_angles = tf.random.uniform(shape=(), minval=-30*np
                                          .pi / 180, maxval=30*np.pi / 180)

    return tfa.image.rotate(image, random_angles), label


def lpf(wave, fs, fe, n):
    """
    # ローパスフィルタ
    wave:入力信号
    fs:周波数
    fe:指定以下の周波数を通過
    n:フィルタの時数
    """
    nyq = fs / 2.0
    b, a = signal.butter(1, fe/nyq, btype='low')
    for i in range(0, n):
        wave = signal.filtfilt(b, a, wave)
    return wave


def weblet_pywt(sig, fs=5):
    wavelet_type = 'cmor1.5-1.0'  # マザーウェーブ
    dt = 1/fs  # サンプリング間隔
    nq_f = fs/2.0  # ナイキスト周波数
    nq_f = 0.8

    # 解析したい周波数のリスト（ナイキスト周波数以下）
    # 1 Hz ～ nq_f Hzの間を等間隔に50分割
    freqs = np.linspace(0.01, nq_f, 250)

    # サンプリング周波数に対する比率を算出
    freqs_rate = freqs / fs

    # スケール：サンプリング周波数＝1:fs(1/dt)としてスケールに換算
    scales = 1 / freqs_rate
    # 逆順に入れ替え
    scale = scales[::-1]

    # weblet
    coef, _ = pywt.cwt(sig, scales=scale, wavelet=wavelet_type)
    t = np.arange(len(sig))/fs
    frq = pywt.scale2frequency(scale=scale, wavelet=wavelet_type)*fs

    return t, frq, coef


def weblet_swan(y, freqs, Fs, omega0=1):
    # ウェーブレット変換
    weblet = pycwt.cwt_f(y, freqs, Fs, pycwt.Morlet(omega0))
    weblet_p = np.abs(weblet)

    return weblet_p


def load_and_preprocess_image(path):
    # ファイルを１つ読み込む。ファイルを表現した生データのテンソルが得られる
    image = tf.io.read_file(path)
    # 生データのテンソルを画像のテンソルに変換する。
    # これによりshape=(240,240,3)、dtype=uint8になる
    image = tf.image.decode_jpeg(image, channels=channel)
    # モデルに合わせてリサイズする
    image = tf.image.resize(image, [highest, width])
    # モデルに合わせて正規化する（値を0〜1の範囲に収める処理）
    image /= 255.0
    return image


def build_model_1d():
    with mirrored_strategy.scope():

        # パラメータの設定
        adabelief = tfa.optimizers.AdaBelief(learning_rate=1e-3, epsilon=1e-16)
        opt = tf.keras.mixed_precision.LossScaleOptimizer(adabelief)
        he = he = tf.keras.initializers.HeNormal()
        l2 = tf.keras.regularizers.l2(0.001)

        # Functional APIでモデルを定義
        inputs = layers.Input(shape=(time, 1))

        # conv層
        x = layers.Conv1D(32, 5, padding='same',
                          kernel_initializer=he, activation=tfa.activations.rrelu)(inputs)
        x = layers.MaxPool1D()(x)

        x = layers.Conv1D(64, 5, padding='same',
                          kernel_initializer=he, activation=tfa.activations.rrelu)(x)
        x = layers.MaxPool1D()(x)

        x = layers.Conv1D(128, 5, padding='same',
                          kernel_initializer=he, activation=tfa.activations.rrelu)(x)
        x = layers.MaxPool1D()(x)

        x = layers.Conv1D(256, 5, padding='same',
                          kernel_initializer=he, activation=tfa.activations.rrelu)(x)
        x = layers.MaxPool1D()(x)

        x = layers.Conv1D(512, 5, padding='same',
                          kernel_initializer=he, activation=tfa.activations.rrelu)(x)

        # 全結合層の定義
        x = layers.LSTM(512, kernel_regularizer=l2)(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(n_classes, activation='softmax')(x)

        # 入出力を定義
        model = tf.keras.models.Model(
            inputs=inputs, outputs=outputs, name='functional')

        # モデルの構築
        model.compile(optimizer=opt, loss='categorical_crossentropy',
                      metrics=['accuracy'])

    model.summary()
    return model


def build_model_2d():
    with mirrored_strategy.scope():

        # パラメータの設定
        adabelief = tfa.optimizers.AdaBelief(learning_rate=1e-3, epsilon=1e-14)
        opt = tf.keras.mixed_precision.LossScaleOptimizer(adabelief)
        he = he = tf.keras.initializers.HeNormal()
        l2 = tf.keras.regularizers.l2(0.001)

        # Functional APIでモデルを定義
        inputs = layers.Input(shape=(highest, width, channel))

        # conv層
        x = layers.Conv2D(32, (3, 3), padding='same',
                          kernel_initializer=he)(inputs)
        x = layers.BatchNormalization()(x)
        x = FReLU(x)
        x = layers.Conv2D(32, (5, 5), padding='same', kernel_initializer=he)(x)
        x = layers.BatchNormalization()(x)
        x = FReLU(x)
        x = layers.MaxPool2D()(x)

        x = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer=he)(x)
        x = layers.BatchNormalization()(x)
        x = FReLU(x)
        x = layers.Conv2D(64, (5, 5), padding='same', kernel_initializer=he)(x)
        x = layers.BatchNormalization()(x)
        x = FReLU(x)
        x = layers.MaxPool2D()(x)

        x = layers.Conv2D(128, (3, 3), padding='same',
                          kernel_initializer=he)(x)
        x = layers.BatchNormalization()(x)
        x = FReLU(x)
        x = layers.Conv2D(128, (5, 5), padding='same',
                          kernel_initializer=he)(x)
        x = layers.BatchNormalization()(x)
        x = FReLU(x)
        x = layers.MaxPool2D()(x)

        x = layers.Conv2D(256, (3, 3), padding='same',
                          kernel_initializer=he)(x)
        x = layers.BatchNormalization()(x)
        x = FReLU(x)
        x = layers.Conv2D(256, (5, 5), padding='same',
                          kernel_initializer=he)(x)
        x = layers.BatchNormalization()(x)
        x = FReLU(x)

        # Attenttion_module
        x = ChannelAttention_Module(x)
        x = SpatialAttention_Module(x)

        # 全結合層の定義
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation=tfa.activations.rrelu,
                         kernel_initializer=he, kernel_regularizer=l2)(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(n_classes, activation='softmax')(x)

        # 入出力を定義
        model = tf.keras.models.Model(
            inputs=inputs, outputs=outputs, name='functional')

        # モデルの構築
        model.compile(optimizer=opt, loss='categorical_crossentropy',
                      metrics=['accuracy'])

    model.summary()
    return model


# log保存場所の設定
root_logdir = os.path.join(os.curdir, "my_logs")


# Tensorbord用のlog
def get_run_logdir(n_classes, psy_index):
    import time
    run_id = time.strftime(psy_index + str(n_classes)+"-run_%Y_%m_%d-%Hh_%Mm")
    return os.path.join(root_logdir, run_id)


def main():
    """
    # mainのプログラム\n
    idは被験者の番号\n
    nは9000をtime(weblet1枚当たりの時間)で割ったもの
    """
    # label, nstの読み込み
    nst_df = pd.read_excel(
        "C://Masato//dataset//psychological_dataset_90s//"+psy_index+"_data//"+psy_index+"_nst.xlsx")
    label_df = pd.read_excel(
        "C://Masato//dataset//psychological_dataset_90s//"+psy_index+"_data//"+psy_index+"_label.xlsx")

    nst_len = len(nst_df.iloc[0])  # 被験者数48
    label_len = len(label_df.iloc[0])  # 被験者数48

    # nstとlabelのデータ数を確認
    if nst_len == label_len:
        print(psy_index+'_load_done:'+str(label_len))
    else:
        print('error!')
        os._exit

    # 被験者48人分のnst,labelを格納
    nst = []
    label = []
    for id in range(nst_len):
        nst.append(nst_df.iloc[:, id].values)
        label.append(label_df.iloc[:, id].values)

    """
    # 一人分[ID1_am]の時系列データを表示
    print(label[0])
    plt.plot(nst[0])
    plt.show()
    plt.cla()

    # weblet画像の作成
    for id in range(nst_len):
        wave = nst[id]
        wave_filted = lpf(wave=wave, fs=5, fe=1, n=5)  # ローパスフィルタ
        t, frq, coef = weblet_pywt(wave_filted)
        for n in range(20):
            # wave = nst[id][n*time:n*time+time]
            # wave_filted = lpf(wave=wave, fs=5, fe=1, n=5)  # ローパスフィルタ
            # t, frq, coef = weblet_data(wave_filted)
            # weblet画像の保存
            plt.pcolormesh(t, frq, np.abs(coef / 255.0), cmap='jet')
            plt.xlabel("Time[s]")
            plt.ylabel("Frequency[Hz]")
            plt.axis('off')  # 軸の削除
            plt.ylim(0.01, 0.06)  # 低周波のみのplot
            plt.xlim(n*time, n*time+time)
            plt.savefig(os.curdir + "//"+psy_index+"_weblet//id" +
                        str(id)+"_weblet"+str(n)+".jpg", bbox_inches='tight', pad_inches=0)
            plt.cla()
    """

    for id in range(nst_len):
        wave = nst[id]
        wave_filted = lpf(wave=wave, fs=5, fe=1, n=5)  # ローパスフィルタ
        freqs = np.arange(0.001, 0.06, 0.001)  # 解析する周波数
        t = np.arange(0, 9000, 1)  # 時間軸
        weblet_p = weblet_swan(y=wave_filted, freqs=freqs, Fs=5)
        for n in range(20):
            if n == 0:
                plt.pcolormesh(
                    t, freqs, 10*np.log(weblet_p), cmap='jet')
                plt.xlabel("Time[s]")
                plt.ylabel("Frequency[Hz]")
                plt.axis('off')  # 軸の削除
                plt.xlim(n*time+10, n*time+time)  # 初期はノイズがあるようなので、10秒スキップ
                plt.savefig(os.curdir + "//"+psy_index+"_weblet//id" +
                            str(id)+"_weblet"+str(n)+".jpg", bbox_inches='tight', pad_inches=0)
                plt.cla()
            else:
                plt.pcolormesh(
                    t, freqs, 10*np.log(weblet_p), cmap='jet')
                plt.xlabel("Time[s]")
                plt.ylabel("Frequency[Hz]")
                plt.axis('off')  # 軸の削除
                plt.xlim(n*time, n*time+time)
                plt.savefig(os.curdir + "//"+psy_index+"_weblet//id" +
                            str(id)+"_weblet"+str(n)+".jpg", bbox_inches='tight', pad_inches=0)
                plt.cla()

    # 入力用のデータを作成
    x_time = []
    x_img = []
    x = []
    y = []
    for id in range(nst_len):
        for n in range(20):
            """
            # 900行ずつnstを読み込む
            # x.append(nst_df.iloc[n*time:n*time+time, id].values) #フィルタをかけずに学習
            x_time.append(
                lpf(wave=nst[id][n*time:n*time+time], fs=5, fe=1, n=5))
            """
            img_path = str(os.curdir + "//"+psy_index +
                           "_weblet//id"+str(id)+"_weblet"+str(n)+".jpg")
            x_img.append(load_and_preprocess_image(img_path))
            # y.append(label[id][n]) #3分類せずにそのまま入力する場合
            # labelを3段階に分類
            if psy_index == 'con':
                if label[id][n] >= 7:
                    y.append(0)  # 集中している
                elif label[id][n] <= 6 and label[id][n] >= 5:
                    y.append(1)  # 少し集中
                else:
                    y.append(2)  # 注意散漫
            elif psy_index == 'drow':
                if label[id][n] <= 3:
                    y.append(0)  # 眠くない
                elif label[id][n] >= 4 and label[id][n] <= 6:
                    y.append(1)  # 少し眠い
                else:
                    y.append(2)  # 眠い
            else:
                if label[id][n] <= 3:
                    y.append(0)  # 疲れていない
                elif label[id][n] >= 4 and label[id][n] <= 5:
                    y.append(1)  # 少し疲れている
                else:
                    y.append(2)  # 疲れている

    # 各labelの総数
    print("label_0: "+str(y.count(0)))
    print("label_1: "+str(y.count(1)))
    print("label_2: "+str(y.count(2)))

    # xの調整
    x = np.array(x_img, dtype=float)
    print(x.shape)
    # yの調整
    y = np.array(y, dtype=float)
    y = y.reshape(y.shape[0], 1)
    print(y.shape)

    y = keras_utils.to_categorical(y, n_classes)  # one-hotに変換
    # print(x, y)

    # 検証データ作成
    X_train, X_valid, Y_train, Y_valid = train_test_split(
        x, y, test_size=0.2, random_state=1)

    # tf datasetに変換
    train_data = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    val_data = tf.data.Dataset.from_tensor_slices((X_valid, Y_valid))

    # batch_sizeで読み込み、プリフェッチ,データ拡張
    train_data = train_data.batch(BATCH_SIZE,
                                  num_parallel_calls=AUTOTUNE).prefetch(buffer_size=AUTOTUNE).map(flip_left_right).shuffle(BATCH_SIZE).repeat(5)
    val_data = val_data.batch(BATCH_SIZE,
                              num_parallel_calls=AUTOTUNE).prefetch(buffer_size=AUTOTUNE)

    """
    # pick images
    plt.figure(figsize=(10, 10), facecolor="white")
    for b_img, b_label in train_data:
        for i, img, label in zip(range(16), b_img, b_label):
            plt.subplot(4, 4, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(img)
        break
    plt.show()
    """

    # Disable AutoShard
    train_data = train_data.with_options(options)
    val_data = val_data.with_options(options)

    # Tensorbord用のlog
    run_logdir = get_run_logdir(n_classes, psy_index)
    # tensorbordの設定
    tensorbord_cb = TensorBoard(log_dir=run_logdir, write_images=True)

    # val_lossの改善が5エポック見られなかったら、学習率を0.5倍する。
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.75,
        patience=3,
        min_lr=1e-9
    )
    # 早期うち止めの設定
    es_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='auto',
        patience=20,
        verbose=1,
        restore_best_weights=True
    )

    # 機械学習
    model = build_model_2d()
    history = model.fit(train_data, validation_data=val_data,
                        epochs=EPOCH, batch_size=BATCH_SIZE,  callbacks=[tensorbord_cb, es_cb, reduce_lr], verbose=1)
    # モデルの評価
    with open(os.curdir+'//'+psy_index+'_temp.txt', 'w')as f:
        score = print(model.evaluate(val_data,
                                     batch_size=BATCH_SIZE, verbose=1), file=f)


if __name__ == '__main__':
    psy_index = 'con'
    main()
    psy_index = 'drow'
    main()
    psy_index = 'fat'
    main()
