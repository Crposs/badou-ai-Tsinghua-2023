from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

img = cv2.imread("lenna.png")
h, w = img.shape[:2]  # 获取图片的high和wide
img_gray = np.zeros([h, w], img.dtype)  # 创建一张和当前图片大小一样的单通道图片
for i in range(h):
    for j in range(w):
        m = img[i, j]  # 取出当前high和wide中的BGR坐标
        img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)  # 将BGR坐标转化为gray坐标并赋值给新图像

print("img show gray: \n %s" % img_gray)
cv2.imshow("img show lenna", img)
cv2.imshow("img show gray", img_gray)

plt.subplot(221)
img = plt.imread("lenna.png")
plt.imshow(img)
print("---image show lenna----\n%s" % img)

img_gray = rgb2gray(img)
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
print("--imag_gray--\n%s" % img_gray)

plt.subplot(223)
# img_binary = np.where(img_gray >= 0.5, 1, 0)
# plt.imshow(img_binary, cmap='binary')
# print("---img_binary---\n%s" % img_binary)

# 二值化
rows, clos = img_gray.shape
for i in range(rows):
    for j in range(clos):
        if (img_gray[i, j] <= 0.5):
            img_gray[i, j] = 0
        else:
            img_gray[i, j] = 1
img_binary2 = img_gray
# cv2的方法展示二值化图像
cv2.imshow("binary",img_binary2)
cv2.waitKey(0)

# plt.imshow(img_binary, cmap='gray')
plt.imshow(img_binary2, cmap='gray')
# print("---img_binary---\n%s" % img_binary)
print("---img_binary2---\n%s" % img_binary2)
print(img_gray.shape)
plt.show()
