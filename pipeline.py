import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import tensorflow as tf


SEED = 42


class GenerateDataset:
    def __init__(self, file):
        self.file = file
        self.seed = 42
        self.BUFFER_SIZE = 1000
        self.BATCH_SIZE = 32

    def decoupage(self, shape_resize=(512, 512, -1)):
        img = np.fromfile(self.file, dtype=np.uint16)  # open raw file
        # img is just an array. Let's reshape it in square
        img = img.reshape(1024, 1024, -1)
        n_3 = 1024//3
        stents = [None]*9
        for i in range(3):
            # resize all stents images in the same size : (512x512)
            stents[i] = cv2.resize(img[:n_3, n_3*i:n_3*(i+1)],
                                   shape_resize[:2]).reshape(shape_resize)
            stents[i+3] = cv2.resize(img[n_3:n_3*2-50,
                                         n_3*i:n_3*(i+1)], shape_resize[:2]).reshape(shape_resize)
            stents[i+6] = cv2.resize(img[n_3*2-50:, n_3*i:n_3*(i+1)],
                                     shape_resize[:2]).reshape(shape_resize)

        # for i, stent in enumerate(stents):
        #     plt.subplot(3, 3, i+1)
        #     plt.imshow(stent, cmap='gray')
        #     plt.title(f"stent {i+1}")
        #     plt.axis('off')
        # plt.show()

        return stents

    def adjusted_brightness(self, image, delta=0.3):
        return tf.image.adjust_brightness(image, delta)

    def flipped_lr(self, image):
        return tf.image.flip_left_right(image)  # left-right

    def random_contrasted(self, image):
        return tf.image.random_contrast(image, 0.2, 0.5, seed=self.seed)

    def flipped_ud(self, image):
        return tf.image.flip_up_down(image)  # up-down

    def cropped(self, image):
        return tf.image.central_crop(image, central_fraction=0.5)

    def rotated(self, image):
        return tf.image.rot90(image)

    def bruit_gaussien_additif(self, image, sigma=0.50):
        shape = image.shape
        noise = np.random.normal(0, sigma, shape)
        return image + noise

    def process(self):
        # 1. Découpage
        stents_list = self.decoupage()

        stents = tf.data.Dataset.from_tensor_slices(stents_list)

        # 2. Ajout du bruit gaussien
        list_datasets = []
        for i in range(100):
            dataset = stents.map(lambda x: (
                self.bruit_gaussien_additif(x, sigma=1+i/100), x))
            list_datasets.append(dataset)
        final_dataset = list_datasets[0]
        for set in list_datasets[1:]:
            final_dataset = final_dataset.concatenate(set)
        # 3. Retourner de gauche à droite
        # dataset = dataset.concatenate(dataset.map(
        #     lambda x, y: (self.flipped_lr(x), y)))
        # 4. Régler luminosité
        # dataset = dataset.concatenate(dataset.map(
        #     lambda x, y: (self.adjusted_brightness(x), y)))
        # 5. Contraste aléatoire
        # dataset = dataset.concatenate(dataset.map(
        #     lambda x, y: (self.random_contrasted(x), y)))
        # 6. Retourner du haut vers le bas
        # dataset = dataset.concatenate(dataset.map(
        #     lambda x, y: (self.flipped_ud(x), y)))
        # 7. otation de 90 degrés
        # dataset = dataset.concatenate(
        #     dataset.map(lambda x, y: (self.rotated(x), y)))
        # 8. Zoom sur l'image
        #! Change le shape de l'image
        # dataset = dataset.concatenate(
        #     dataset.map(lambda x, y: (self.cropped(x), y)))
        # 9. Mettre les pixels entre 0 et 1
        final_dataset = final_dataset.map(lambda x, y: (
            x/tf.reduce_max(x), y/tf.reduce_max(y)))

        return final_dataset

    # TODO fonction get augmented d'Adrien


class Pipeline(GenerateDataset):
    def __init__(self, file):
        super().__init__(file)
        self.dataset = self.process()

        #! le mieux serait d'avoir getaugmented en classe
        # self.dataset = (
        #     self.dataset
        #     .cache()
        #     .shuffle(self.BUFFER_SIZE)
        #     .batch(self.BATCH_SIZE)
        #     .repeat()
        #     # .map(GetAugmented)
        #     .prefetch(buffer_size=tf.data.AUTOTUNE)
        # )
        # self.dataset = self.dataset.batch(self.BATCH_SIZE)


# dataset = Pipeline(file='CDStent.raw').dataset


def vizualize_pipeline_dataset():
    dataset = Pipeline(file='CDStent.raw').dataset
    print(len(dataset))
    # for batch_num, batch_instances in enumerate(list(dataset.batch(72).as_numpy_iterator())):
    #     fig, axes = plt.subplots(9, 8)
    #     batch_num += 1
    #     axes = axes.ravel()
    #     (noised_images, labels) = batch_instances
    #     plt.suptitle(f'Batch {batch_num}')
    #     for i, img in enumerate(noised_images):
    #         axes[i].imshow(img, cmap='gray')
    #         axes[i].axis('off')
    #     # plt.savefig(f'Results/dataset_batch_{batch_num}.png')
    #     plt.show()


if __name__ == "__main__":
    vizualize_pipeline_dataset()
