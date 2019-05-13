import matplotlib.pyplot as plt
import numpy as np
from skimage import color, feature


img = plt.imread('./img/moneda4.jpg')
gris = color.rgb2gray(img)

plt.imshow(img)
plt.figure()
plt.imshow(gris, cmap='gray')

from skimage.filters import threshold_li

binarizado = threshold_li(gris)
print('Selected a threshold of %.2f' % binarizado)
mascara = gris > binarizado
plt.imshow(mascara, cmap='gray');

from skimage import morphology
from skimage.morphology import disk

mascara_limpia = morphology.remove_small_objects(mascara)
mascara_limpia = ~morphology.remove_small_objects(~mascara)

plt.imshow(mascara_limpia, cmap='gray');

#plt.imshow(coins, cmap='gray');

mascara_de_fondo = ~mascara_limpia

img.setflags(write=1)
img[mascara_de_fondo] = 0
gris[mascara_de_fondo] = 0
plt.imshow(img);

from scipy import ndimage as ndi
from matplotlib.colors import ListedColormap

distancia_imagen = ndi.distance_transform_edt(mascara_de_fondo)
print('transformada de distancia:', distancia_imagen.shape, distancia_imagen.dtype)

from skimage import feature, measure

def imshow_overlay(im, mask, alpha=0.5, color='red', **kwargs):
    """Show semi-transparent red mask over an image"""
    mask = mask > 0
    mask = np.ma.masked_where(~mask, mask)        
    plt.imshow(im, **kwargs)
    plt.imshow(mask, alpha=alpha, cmap=ListedColormap([color]))


def watershed_segmentation(mask):
    distancia_imagen = ndi.distance_transform_edt(mask)
    peaks = feature.peak_local_max(distancia_imagen, indices=True)
    peaks_im = np.zeros(distancia_imagen.shape, dtype=bool)#matriz de zeros de la mascara
    for row, col in peaks:
        peaks_im[row, col] = 1
    marcadores_imagen = measure.label(peaks_im)
    etiqueta_imagen = morphology.watershed(-distancia_imagen, marcadores_imagen, mask=mascara_limpia)
    return etiqueta_imagen


etiqueta_moneda = watershed_segmentation(mascara_limpia)

plt.imshow(etiqueta_moneda)

regions = measure.regionprops(etiqueta_moneda)

plt.imshow(img)

for region in regions:
    y, x = region.centroid
    area = region.area
    area_str = '%.1f' % (area/100)

min_5 = 61000
max_5=62000
min_10 = 70000
max_2=51000
min_2=48000
max_1=42000
min_1=39000


num_5 = 0
num_10 = 0
num_deruido = 0
num_2 = 0
num_1=0



for region in regions:
    y, x = region.centroid
    area = region.area
    if area >= min_5 and area<= max_5:
        coin_name = '5'
        num_5 += 1
    elif area >= min_10:
        coin_name = '10'
        num_10 += 1
    elif area >= min_2 and area<=max_2:
        coin_name = '2'
        num_2 += 1
    elif area >= min_1 and area<=max_1:
        coin_name = '1'
        num_1 += 1
    else:
        coin_name = ''
        num_deruido += 1
    plt.text(x, y, coin_name, ha='center', va='center')  # ha, va = horizontal alignment,vertical aligment
    
value = (10*num_10 + 5*num_5 + 2*num_2+1*num_1)
monedas=(num_10+num_5+num_2+num_1)
print ('%i :10 pesos, %i: 5 pesos, %i :2 pesos, %i :1 peso' %(num_10, num_5, num_2, num_1))
print ('Tienes $%.2f pesos' % value)
print('Numero de monedas: ',monedas)

plt.show()