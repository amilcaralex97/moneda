import matplotlib.pyplot as plt
import numpy as np
from skimage import color, feature


img = plt.imread('./img/moneda4.jpg')
gray_im = color.rgb2gray(img)

plt.imshow(img)
plt.figure()
plt.imshow(gray_im, cmap='gray')

from skimage.filters import threshold_li

coin_thresh = threshold_li(gray_im)
print('Selected a threshold of %.2f' % coin_thresh)
coin_mask = gray_im > coin_thresh
plt.imshow(coin_mask, cmap='gray');

from skimage import morphology
from skimage.morphology import disk

coin_mask_clean = morphology.remove_small_objects(coin_mask)
coin_mask_clean = ~morphology.remove_small_objects(~coin_mask_clean)

plt.imshow(coin_mask_clean, cmap='gray');

''' no_small = morphology.remove_small_objects(coin_mask, min_size=150)

coins = morphology.binary_closing(no_small,disk(3)) '''

#plt.imshow(coins, cmap='gray');

bg_mask = ~coin_mask_clean

img.setflags(write=1)
img[bg_mask] = 0
gray_im[bg_mask] = 0
plt.imshow(img);

''' im.setflags(write=1)
gray_im.setflags(write=1)
im[coins==False] = 0
gray_im[coins==False] = 0 '''

from scipy import ndimage as ndi
from matplotlib.colors import ListedColormap

distance_im = ndi.distance_transform_edt(bg_mask)
print('distance transform:', distance_im.shape, distance_im.dtype)

from skimage import feature, measure

def imshow_overlay(im, mask, alpha=0.5, color='red', **kwargs):
    """Show semi-transparent red mask over an image"""
    mask = mask > 0
    mask = np.ma.masked_where(~mask, mask)        
    plt.imshow(im, **kwargs)
    plt.imshow(mask, alpha=alpha, cmap=ListedColormap([color]))


def watershed_segmentation(mask):
    distance_im = ndi.distance_transform_edt(mask)
    peaks = feature.peak_local_max(distance_im, indices=True)
    peaks_im = np.zeros(distance_im.shape, dtype=bool)
    for row, col in peaks:
        peaks_im[row, col] = 1
    markers_im = measure.label(peaks_im)
    labelled_im = morphology.watershed(-distance_im, markers_im, mask=coin_mask_clean)
    return labelled_im


labelled_coin_im = watershed_segmentation(coin_mask_clean)

print ('There are %i coins in the image.' % (len(np.unique(labelled_coin_im)) - 1))

plt.imshow(labelled_coin_im)

regions = measure.regionprops(labelled_coin_im)

plt.imshow(img)

for region in regions:
    y, x = region.centroid
    area = region.area
    area_str = '%.1f' % (area/100)
    plt.text(x, y, area_str, color='k', ha='center', va='center')

plt.show()