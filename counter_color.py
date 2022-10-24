import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import sobel, threshold_otsu, threshold_local
from skimage.measure import label, regionprops
from skimage.morphology import *
from collections import defaultdict
from skimage import color

inside_corners = [np.array([[1, 1], [1, 0]]),
                  np.array([[1, 1], [0, 1]]),
                  np.array([[1, 0], [1, 1]]),
                  np.array([[0, 1], [1, 1]]),]

def is_react(prop) -> bool:
    image = prop.image
    ic = 0

    for y in range(0, image.shape[0] - 1):
        for x in range(0, image.shape[1] - 1):
            sub = image[y : y + 2, x : x + 2]
            for mask in inside_corners:
                if np.all(sub == mask):
                    ic += 1
                    break

    return ic == 4

def define_color(color, colors, delta):
    for elem in colors:
        if abs(color - elem) < delta:
            return elem
    
    raise "Color not defined"

source_image = plt.imread("balls_and_rects.png")
hsv = color.rgb2hsv(source_image)[ : , : , 0]

plt.figure("original, hsv, binary and labeled")
plt.subplot(141)
plt.imshow(source_image)
plt.subplot(142)
plt.imshow(hsv)

bin_image = source_image.mean(2)
bin_image[bin_image > 0] = 1

plt.subplot(143)
plt.imshow(bin_image)

labeled , total_figures = label(bin_image, return_num=True)

print(f"Всего фигур: {total_figures}")

plt.subplot(144)
plt.imshow(labeled)

props = regionprops(labeled)
colors_list = []

for prop in props:
    cy, cx = prop.centroid
    color = hsv[int(cy), int(cx)]
    colors_list += [color]

colors_list.sort()
diff = np.diff(colors_list)
delta = np.std(diff) * 2

colors = []
quantity_colors = defaultdict(lambda : 0)
now_group_color = - 100
for color in colors_list:
    if abs(color - now_group_color) > delta:
        now_group_color = color
        colors += [now_group_color]
    quantity_colors[now_group_color] += 1

react_colors = defaultdict(lambda : 0)
circle_colors = defaultdict(lambda : 0)

for prop in props:
    cy, cx = prop.centroid

    color = define_color(hsv[int(cy), int(cx)], colors, delta)

    if np.all(prop.image) == 1:
        react_colors[color] += 1
    else:
        circle_colors[color] += 1

print("Список цветов и количество повторений:")
react_quantity = 0
for color in colors:
    print(f"Цвет: {color} - > {quantity_colors[color]}. Reacts: {react_colors[color]}; Circles: {circle_colors[color]}.")
    react_quantity += react_colors[color]

print(f"-------\nВсего цветов: {len(colors)}")
print(f"Всего фигур: {total_figures}. Прямоугольников -> {react_quantity}; Кругов -> {total_figures - react_quantity}.")

plt.figure()
plt.plot(colors_list)
plt.show()