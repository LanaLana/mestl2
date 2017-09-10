# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot
import math
from skimage import measure 
from scipy.spatial import distance
from scipy.spatial import ConvexHull 

def segment_line_bilder_try(image, check=False, blur=False, porog=0, Thresh213=False):
    img = cv2.imread(image)     #открываем изображение
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #изображение цветное
    
    if check == True:
        matplotlib.pyplot.imshow(img)
        matplotlib.pyplot.axis('off')
        matplotlib.pyplot.show()
   
    #серое изображение
    if blur:
        gs = cv2.cvtColor(cv2.medianBlur(img, 5), cv2.COLOR_RGB2GRAY)
    else:
        gs = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)   
    
    #бинаризация
    #ret, th = cv2.threshold(gs, 80, 255, cv2.THRESH_BINARY)
    if not Thresh213:
        th = cv2.adaptiveThreshold(gs, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 2 ** 12 - 1, porog) 
    else:
        th = cv2.adaptiveThreshold(gs, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 2 ** 13 - 1, porog)
    
    if check == True:
        matplotlib.pyplot.imshow(th, 'gray')
        matplotlib.pyplot.axis('off')
        matplotlib.pyplot.show()
    
    #выделение компонент связности
    L = measure.label(th)
    components = np.max(L) 
    # np.unique(L)
    # >> array([0, 1, 2]) у каждой компоненты свой номер пикселей
    
    # удаление компонент связности, не являющихся ладонью
    if components > 1:
        a = np.bincount(L.ravel())
        max_1 = np.argmax(a)
        a[max_1] = -1
        max_2 = np.argmax(a)
        for i in range(1, len(np.unique(L))):
            if i != max_1 and i != max_2:
                mask = (L == i)
                th[mask] = 0 
     
    if check == True:
        matplotlib.pyplot.imshow(th, 'gray')
        matplotlib.pyplot.axis('off')
        matplotlib.pyplot.show()
    
    # эрозия и дилатация для сжатия ладони в "кулак"
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(71,71))
    erosion = cv2.erode(th, kernel) 
    dilation = cv2.dilate(erosion, kernel)
    
    if check == True:
        matplotlib.pyplot.imshow(dilation, 'gray')
        matplotlib.pyplot.axis('off')
        matplotlib.pyplot.show()
    
    # вычитание из исходного изображения кулака
    fingers = th & ~dilation
    
    if check == True:
        matplotlib.pyplot.imshow(fingers, 'gray')
        matplotlib.pyplot.axis('off')
        matplotlib.pyplot.show()
    
    # удаление шума
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    erosion = cv2.erode(fingers, kernel)  
    fingers_denois = cv2.dilate(erosion, kernel)
    
    #удаление некомпактных
    L = measure.label(fingers_denois)
    tmp = measure.regionprops(L, intensity_image=None, cache=True)
    for i in tmp:
        comp = (i.perimeter ** 2) / i.coords.shape[0]
        if (comp < 16) or (comp > 40):
            for c in i.coords:
                fingers_denois[c[0], c[1]] = 0
    
    #выделение компонент связности
    L = measure.label(fingers_denois)
    components = np.max(L)
    # np.unique(L)
    # >> array([0, 1, 2]) у каждой компоненты свой номер пикселей
    
    # удаление компонент связности, не являющихся пальцами
    
    if components > 5:
        a = np.bincount(L.ravel())
        for k in range(components - 5):
            max_ = np.argmin(a)
            a[max_] = max(a) + 1
            fingers_denois[L == max_] = 0
    
    if components < 5:
        raise "Error"
    
    if check == True:
        matplotlib.pyplot.imshow(fingers_denois, 'gray')
        matplotlib.pyplot.axis('off')
        matplotlib.pyplot.show()
    
    # определение компонент связности - пальцев
    L1 = measure.label(fingers_denois)
    components = np.max(L1)  
    
    # построение главной оси каждого пальца
    tmp = measure.regionprops(L1, intensity_image=None, cache=True)
    fing_points = {}
    j = 0
    for i in range(len(tmp)):
        axis_len = tmp[i].major_axis_length
        orientat = tmp[i].orientation
        fing_points[j] = [tmp[i].centroid[1], tmp[i].centroid[0]]
        fing_points[j + 1] = [tmp[i].centroid[1], tmp[i].centroid[0]]
        while fingers_denois[int(fing_points[j][1]), int(fing_points[j][0])]:
            fing_points[j][0] -= math.cos(orientat)
            fing_points[j][1] += math.sin(orientat)
        while fingers_denois[int(fing_points[j + 1][1]), int(fing_points[j + 1][0])]:
            fing_points[j + 1][0] += math.cos(orientat)
            fing_points[j + 1][1] -= math.sin(orientat)
        j = j + 2
        
    # проставление вершин ломанной на изображении
    j = 0
    img_points = cv2.cvtColor(fingers_denois, cv2.COLOR_GRAY2RGB) 
    for i in range(len(tmp)):
        cv2.circle(img_points, (int(fing_points[j][0]), int(fing_points[j][1])), 5, (0, 255, 0), -1)
        cv2.circle(img_points, (int(fing_points[j + 1][0]), int(fing_points[j + 1][1])), 5, (0, 255, 0), -1)
        j = j + 2
    
    if check == True:
        matplotlib.pyplot.imshow(img_points)
        matplotlib.pyplot.axis('off')
        matplotlib.pyplot.show()
    
    #определение центра ладони
    L2 = measure.label(dilation)   
    kulak = measure.regionprops(L2, intensity_image=None, cache=True)
    kulak_points = [kulak[0].centroid[1], kulak[0].centroid[0]]
    img_kulak = cv2.cvtColor(dilation, cv2.COLOR_GRAY2RGB) 
    cv2.circle(img_kulak, (int(kulak_points[0]), int(kulak_points[1])), 5, (0, 255, 0), -1) 
    
    if check == True:
        matplotlib.pyplot.imshow(img_kulak)
        matplotlib.pyplot.axis('off')
        matplotlib.pyplot.show()
    
    #определение прохода пальцев для ломанной
    kulak_fing = []
    finger_vershinki = []
    orient = []
    minor_axis = []
    persquare = []
    for i in range(len(fing_points)/2):
        orient.append(tmp[i].orientation)
        minor_axis.append(tmp[i].minor_axis_length)
        persquare.append((tmp[i].perimeter, tmp[i].coords.shape[0]))
        if distance.euclidean(fing_points[i * 2], kulak_points) > distance.euclidean(fing_points[i * 2 + 1], 
                                                                                     kulak_points):
            kulak_fing.append(fing_points[i * 2 + 1])
            finger_vershinki.append(fing_points[i * 2])
        else:
            kulak_fing.append(fing_points[i * 2]) 
            finger_vershinki.append(fing_points[i * 2 + 1])
            
    # выпуклая оболочка
    hull = ConvexHull(kulak_fing)
    #print hull.vertices
    # >> [4 2 0 1 3]
    
    
    two_points = []
    for i in range(len(kulak_fing)):
        u = kulak_fing[hull.vertices[i]]
        v = kulak_fing[hull.vertices[(i + 1) % len(kulak_fing)]]
        two_points.append(distance.euclidean(u, v))
    st = (np.argmax(two_points) + 1) % len(kulak_fing)
    vershinki = []
    low_point = []
    seredinki = []
    ugolochki = []
    minor_axis_sort = []
    persquare_sort = []
    for i in range(len(kulak_fing) - 1):
        u = kulak_fing[hull.vertices[(i + st) % len(kulak_fing)]]
        v = kulak_fing[hull.vertices[(i + st + 1) % len(kulak_fing)]]
        seredinki.append((np.array(u) + np.array(v)) / 2.0)
    for i in range(len(kulak_fing)):
        v = finger_vershinki[hull.vertices[(i + st) % len(kulak_fing)]]
        u = kulak_fing[hull.vertices[(i + st) % len(kulak_fing)]]
        vershinki.append(np.array(v))
        low_point.append(np.array(u))
        ugolochki.append(orient[hull.vertices[(i + st) % len(kulak_fing)]])
        minor_axis_sort.append(minor_axis[hull.vertices[(i + st) % len(kulak_fing)]])
        persquare_sort.append(persquare[hull.vertices[(i + st) % len(kulak_fing)]])
        
    if (distance.euclidean(seredinki[0], seredinki[1]) < distance.euclidean(seredinki[-1], seredinki[-2])):
        seredinki = list(reversed(seredinki))
        vershinki = list(reversed(vershinki))
        low_point = list(reversed(low_point))
        ugolochki = list(reversed(ugolochki))
        minor_axis_sort = list(reversed(minor_axis_sort))
        persquare_sort = list(reversed(persquare_sort))
    
    # двигаемся к кулаку
    for i in range(1, len(seredinki)):
        r = np.array(kulak_points) - seredinki[i]
        r /= np.linalg.norm(r)
        while (not th[int(seredinki[i][1]), int(seredinki[i][0])]):
            seredinki[i] += r
            
    seredinki_1 = low_point[0] - minor_axis_sort[0] * np.array([np.sin(ugolochki[0]), np.cos(ugolochki[0])]) / 2.0
    seredinki_2 = low_point[0] + minor_axis_sort[0] * np.array([np.sin(ugolochki[0]), np.cos(ugolochki[0])]) / 2.0
    if (distance.euclidean(seredinki[1], seredinki_1) < distance.euclidean(seredinki[1], seredinki_2)):
        seredinki[0] = seredinki_1
    else:
        seredinki[0] = seredinki_2
        
    r = np.array([-math.cos(ugolochki[0]), math.sin(ugolochki[0])])
    if distance.euclidean(kulak_points, seredinki[0] - r) < distance.euclidean(kulak_points, seredinki[0] + r):
        r = -r
    while sum(img_kulak[int(seredinki[0][1]), int(seredinki[0][0])]) == 0:
        seredinki[0] += r
    
    # проверка, что нет тупых углов
    #for i in range(4):
    #    angles_between_fing_max[i] = max(angle(vershinki[i], seredinki[i], vershinki[i + 1]), 
    #                                     angles_between_fing_max[i])
    #    angles_between_fing_min[i] = min(angle(vershinki[i], seredinki[i], vershinki[i + 1]), 
    #                                     angles_between_fing_min[i])
    # вспомогательная функция, позволила определить max и min углов между пальцами:
    #angles_between_fing_max = [0.92087624059909257, 0.96625691917013867, 0.97277528218425491, 0.95139379777545141]
    #angles_between_fing_min = [0.008030813304325337, 0.66696613527220039, 0.68061323717560529, 0.42954030001174476]
 
    angles_between_fing_max = [0.95, 0.98, 0.99, 0.97]
    angles_between_fing_min = [-0.1, 0.40, 0.44, 0.22]
    
    angles_between_fing = 4 * [0]
    for i in range(4):
        angles_between_fing[i] = angle(vershinki[i], seredinki[i], vershinki[i + 1])
        if angles_between_fing < angles_between_fing_min and angles_between_fing > angles_between_fing_max:
            throw(1)
    
    linza = []
    for i in range(1, 4):
        linza.append(np.linalg.norm(vershinki[i + 1] - seredinki[i]))
    linza.append(np.linalg.norm(vershinki[1] - seredinki[1]))
    if np.std(linza) > 28:
        throw(1)
    
    line_picture = img.copy()
    for i in range(len(seredinki)):
        cv2.line(line_picture, (int(vershinki[i][0]), int(vershinki[i][1])),
                 (int(seredinki[i][0]), int(seredinki[i][1])), (0, 255, 0), 3)
        cv2.line(line_picture, (int(vershinki[i + 1][0]), int(vershinki[i + 1][1])),
                 (int(seredinki[i][0]), int(seredinki[i][1])), (0, 255, 0), 3)
        
    
    for i in range(len(vershinki)):
        cv2.circle(line_picture, (int(vershinki[i][0]), int(vershinki[i][1])), 5, (255, 255, 255), -1)
    for i in range(len(seredinki)):
        cv2.circle(line_picture, (int(seredinki[i][0]), int(seredinki[i][1])), 5, (255, 255, 255), -1)
   
    if check == True:
        matplotlib.pyplot.imshow(line_picture)
        matplotlib.pyplot.axis('off')
        matplotlib.pyplot.show()
    
    # создание вектора признаков
    feature = np.empty(8, dtype='float64')
    j = 0
    for i in range(4):
        feature[j] = np.linalg.norm(vershinki[i] - seredinki[i])
        j = j + 1
        feature[j] = np.linalg.norm(seredinki[i] - vershinki[i + 1])
        j = j + 1
        
    return line_picture, feature

 
def angle(a, b, c):
    
    ba = np.linalg.norm(a - b)
    bc = np.linalg.norm(c - b)
    
    return np.dot(a - b, c - b) / (ba * bc)

def segment_line_bilder(image, check=False):
    try:
        return segment_line_bilder_try(image, check=check, blur=True)
    except:
        pass
    try:
        return segment_line_bilder_try(image, check=check, blur=False)
    except:
        pass
    try:
        return segment_line_bilder_try(image, check=check, blur=True, porog=8)
    except:
        pass
    try:
        return segment_line_bilder_try(image, check=check, Thresh213=True)
    except:
        pass
