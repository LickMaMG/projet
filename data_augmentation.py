## bruit gaussien ##
import numpy as np
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

image = np.fromfile('CDStent.raw', dtype=np.uint16)
image = image.reshape(1024, -1)


def bruit_gaussien_additif(image, n, sigma):
    shape = image.shape
    noise = np.random.normal(0, sigma, shape)
    return image + noise


## test ##
nouvel_image = bruit_gaussien_additif(image, 20, 5)
plt.imshow(nouvel_image, cmap='gray')
plt.title("image bruité")
plt.show()

# test ACP
# svd = TruncatedSVD(n_components=5, n_iter=7)
# image_debruite = svd.fit(nouvel_image)
# plt.imshow(image_debruite, cmap='gray')
# plt.title("image debruité")
# plt.show()
