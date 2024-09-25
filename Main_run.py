import train_config
from model_net import *

from my_metric import *

os.environ['OMP_NUM_THREADS'] = '5'


def load_model():
    try:
        raise OSError
        model1 = keras.models.load_model(model_path)
        return model1
    except OSError:
        model1 = AAUnet()
        model1.compile(loss=None, optimizer=keras.optimizers.Adam(learning_rate=train_config.learning_rate))  # 编译模型 可以在这里指定损失函数、指标、学习率、优化器....
        return model1


if __name__ == '__main__':
    model = load_model()


    callback = [
        keras.callbacks.TensorBoard(
            log_dir=log_path,
            histogram_freq=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=True
        ),
        MetricsCallback(_valid_dataset=plot_dataset, _image_path=image_path)

    ]
    model.summary()

    sample = iter(train_dataset.take(1))


    with tf.device(device):
        if is_train:
            model.fit(x=train_dataset, validation_data=test_dataset, epochs=epoch_size,
                      steps_per_epoch=steps_per_epoch
                      , callbacks=callback, shuffle=False, validation_steps=val_step)
    # model.save('model/model.h5')

