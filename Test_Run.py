import Main_run
import load_dataset
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    model = Main_run.load_model()
    model.load_weights('model/model_139.h5')
    dataset = load_dataset.test_hospital
    data_list = list(iter(dataset))

    sample = model.test_step(data_list[0]).keys()
    list_ = {i: [] for i in sample}

    for i in data_list:
        per_test = model.test_step(i)
        print('....')
        for key, value in per_test.items():
            list_[key].append(value)

    for key, value in list_.items():
        print(key, np.mean(value))

    data_list = list(iter(dataset.map(lambda x, _: x)))
    pred_list = [model.predict(i) for i in data_list]

    index = 0

    for ori_batch, p_mask_batch in zip(data_list, pred_list):
        for o_image, o_mask, p_mask in zip(ori_batch[0], ori_batch[1], p_mask_batch):
            total_pixels = 128 * 128
            num_of_ones = tf.reduce_sum(o_mask)
            ratio = num_of_ones / total_pixels
            ratio = "{:.5f}".format(ratio)
            fig, _ = plt.subplots(1, 3)
            plt.subplot(1, 3, 1)
            plt.axis('off')
            plt.imshow(o_image)
            plt.subplot(1, 3, 2)
            plt.axis('off')
            plt.imshow(o_mask, cmap='gray')
            plt.subplot(1, 3, 3)
            plt.axis('off')
            p_mask_bin = np.where(p_mask > 0.5, 1.0, 0.0)
            plt.imshow(p_mask_bin, cmap='gray')
            fig.savefig('./test_image/' + f"{index}_true{ratio}.png")

            plt.close(fig)
            index += 1
