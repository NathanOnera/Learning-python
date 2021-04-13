import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageOps
from PIL import ImageFilter
import numpy as np
import cmath
import math
#from scipy import convolve,gaussian_filter
from scipy.ndimage.filters import convolve,gaussian_filter




img_ = Image.open("screen_tft.PNG")
img = img_.convert('L')

mat = np.array(img)*1.0
mat = gaussian_filter(mat,1)

print(mat) 
### Convolution + Gradient
HX = np.array([[-1/8,0,1/8],[-2/8,0,2/8],[-1/8,0,1/8]])
HY = np.array([[-1/8,-2/8,-1/8],[0,0,0],[1/8,2/8,1/8]])

deriveX = convolve(mat,HX)

print(deriveX)

deriveY = convolve(mat,HY)

Grad = deriveX + deriveY*1j

print(Grad)

G = np.absolute(Grad)
Theta = np.angle(Grad)


print(G)

img_G = Image.fromarray(G).convert('L')
img_G_ = ImageOps.autocontrast(img_G)

s = G.shape
seuil = 10
for i in range(s[0]):
    for j in range(s[1]):
        if G[i][j] < seuil:
            G[i][j] = 0.0



img.show()
img_G_.show()

'''
Image.fromarray(G).convert('L').show()
'''

'''
############### Filtre flou
noise = np.random.normal(0,7,mat.shape)

img_bruit = Image.fromarray(mat+noise).convert('L')
img.show()
img_bruit.show()
img_bruit.filter(ImageFilter.BoxBlur(1)).show()
img2 = img.copy()

n,m = img.size
for i in range(n-2):
    for j in range(m-2):
        value = (img.getpixel((i+1,j+1)) + img.getpixel((i,j+1)) + img.getpixel((i+2,j+1)) + img.getpixel((i+1,j+2)) + img.getpixel((i+1,j)) + img.getpixel((i,j)) + img.getpixel((i+2,j+2)) + img.getpixel((i,j+2)) + img.getpixel((i+2,j)))/9
        img2.putpixel((i+1,j+1),math.floor(value))
img2.show()
'''


'''
mat = np.array(img)
print(mat)

n,bins,patches = plt.hist(mat.flatten(),bins = range(256),density=True,cumulative=True)
#plt.show()

img_cor = ImageOps.equalize(img)
img_cor.show()
mat_cor = np.array(img_cor)

n,bins,patches = plt.hist(mat_cor.flatten(),bins = range(256),density=True,cumulative=True)
#plt.show()

img_cor.rotate(45).show()
'''



#img_ori = Image.open("screen_tft.PNG")
#img = img_ori.convert('L')
#w,h = img.size

#print("Largeur : {} px, hauteur : {} px.".format(w,h))
#print("Format des pixels : {}".format(img.mode))
#x = 100
#y = 100
#p_val = img.getpixel((x,y))
#print("Valeur du pixel situé en ({},{}) : {}".format(x,y,p_val))

#mat  = np.array(img)
#print(mat)


