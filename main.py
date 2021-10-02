import shutil
import cv2 as cv
import os
import re

import numpy as np
from scipy import stats
import pickle

import matplotlib.pyplot as plt
import xlsxwriter

from time import process_time

dirname_learn = 'ALL' # ALL is a folder containing all images
dirname_predict = 'Predict_ALL'

# -----------------------------------------------------------
# -----------------------------------------------------------
# Learning

dirname1 = 'Cropped'
if os.path.exists(dirname1):
    shutil.rmtree(dirname1)
os.makedirs(dirname1)
dirname2 = 'Color'
if os.path.exists(dirname2):
    shutil.rmtree(dirname2)
os.makedirs(dirname2)
dirname3 = 'Rotate'
if os.path.exists(dirname3):
    shutil.rmtree(dirname3)
os.makedirs(dirname3)
dirname4 = 'draw'
if os.path.exists(dirname4):
    shutil.rmtree(dirname4)
os.makedirs(dirname4)
dirname5 = 'Big circle'
if os.path.exists(dirname5):
    shutil.rmtree(dirname5)
os.makedirs(dirname5)
dirname6 = 'Contour'
if os.path.exists(dirname6):
    shutil.rmtree(dirname6)
os.makedirs(dirname6)
dirname7 = 'scale'
if os.path.exists(dirname7):
    shutil.rmtree(dirname7)
os.makedirs(dirname7)
dirname9 = 'circle'
if os.path.exists(dirname9):
    shutil.rmtree(dirname9)
os.makedirs(dirname9)

# Step 1
# -------------------------------
img_folder = os.path.join(os.getcwd(), dirname_learn)
img_files = [f for f in os.listdir(img_folder) if f.endswith('.jpg')]
for f in img_files:
    cimg = cv.imread(os.path.join(img_folder, f))
    gray = cv.cvtColor(cimg, cv.COLOR_BGR2GRAY)
    print("processing.... {}".format(f))
    circles = cv.HoughCircles(gray,cv.HOUGH_GRADIENT,10,1800, maxRadius=800, minRadius=50)
    circles = np.uint16(np.around(circles))
    x, y, r = 0, 0, 0
    for i in circles[0,:]:
        x, y, r = i[0], i[1], i[2]
    dst = cimg[y-r:y + r, x-r:x + r]
    cv.imwrite(os.path.join(dirname1, f), dst)

# Step 2
# ---------------------------------------------
# Find the extreme points and Crop the image
# Big one part

img_folder = os.path.join(os.getcwd(), 'Cropped')
img_files = [f for f in os.listdir(img_folder) if f.endswith('.jpg')]

for f in img_files:
    print("Processing Contouring {} ...".format(f))
    img = cv.imread(os.path.join(img_folder, f))
    # (1) Convert to gray, and threshold
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, threshed = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # (2) Morph-op to remove noise
    kernel = np.ones((20, 20), np.uint8)
    morphed = cv.morphologyEx(threshed, cv.MORPH_CLOSE, kernel)

    # (3) Find the max-area contour
    _, cnts, _ = cv.findContours(threshed, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    cnt = sorted(cnts, key=cv.contourArea)[-1]

    # (4) Draw the Extreme Points

    h_img, w_img = img.shape[:2]

    extLeft = tuple(cnt[cnt[:, :, 0].argmin()][0])
    extRight = tuple(cnt[cnt[:, :, 0].argmax()][0])
    extTop = tuple(cnt[cnt[:, :, 1].argmin()][0])
    extBot = tuple(cnt[cnt[:, :, 1].argmax()][0])

    x = extLeft[0]
    w = extRight[0]
    y = extTop[1]
    h = extBot[1]
    dst = img[y: h, x:w-40]
    height, width = dst.shape[:2]

    cv.imwrite(os.path.join(dirname5, f), dst) # Big one circle

# Step 3
# -------------------------------
img_folder = os.path.join(os.getcwd(), 'Big circle')
img_files = [f for f in os.listdir(img_folder) if f.endswith('.jpg')]

for f in img_files:
    print("STEP 2 - Cropping  {} ...".format(f))
    img = cv.imread(os.path.join(img_folder, f))

    x, y, w, h = 350, 200, 450, 330
    dst = img[y:y + h, x:x + w]

    cv.imwrite(os.path.join(dirname2, f), dst) # Color -> shaft line

# # Step 4
# # -----------------------------------------------------
# # Rotate the images from Step 1

img_folder = os.path.join(os.getcwd(), 'Color')
img_files = [f for f in os.listdir(img_folder) if f.endswith('.jpg')]
img_folder1 = os.path.join(os.getcwd(), 'Big circle')
img_files1 = [f for f in os.listdir(img_folder1) if f.endswith('.jpg')]

for f in img_files:
    img = cv.imread(os.path.join(img_folder, f))
    # (1) Convert to gray, and threshold
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, threshed = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # (2) Morph-op to remove noise
    kernel = np.ones((20, 20), np.uint8)
    morphed = cv.morphologyEx(threshed, cv.MORPH_CLOSE, kernel)

    # (3) Find the max-area contour
    _, cnts, _ = cv.findContours(morphed, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    cnts = sorted(cnts, key=cv.contourArea)

    h_img, w_img = img.shape[:2]
    def give_teta(cnt, conter=0):
        pixelpoints = cnt[np.logical_and(20 < cnt[:, 0, 0], cnt[:, 0, 0] < w_img - 45)]
        pixelpoints = pixelpoints[np.logical_and(20 < pixelpoints[:, 0, 1], pixelpoints[:, 0, 1] < h_img - 10)]
        x = pixelpoints[:, :, 0].flatten()
        y = pixelpoints[:, :, 1].flatten()
        cv.drawContours(img, [pixelpoints], 0, (255, 0, 0), 2)
        cv.imwrite(os.path.join(dirname4, f), img)
        slope, _, _, _, _ = stats.linregress(x, y)

        teta = np.arctan(slope)*180/np.pi
        return teta

    cnt = cnts[-1]
    teta1 = give_teta(cnt,1)
    cnt = cnts[-2]
    teta2 = give_teta(cnt)
    # cnt = cnts[-3]
    # teta3 = give_teta(cnt)
    # print(teta1, teta2, teta3)
    # teta4 = give_teta(cnts[-4])
    teta_eq = ((teta1 + teta2)/2)

    print("Rotating {} ...".format(f))
    # print(teta1, " + ", teta2, " + ", teta3, " = ", teta_eq)
    rows, cols = img.shape[:2]

    img1 = cv.imread(os.path.join(img_folder1, f))
    rows, cols = img1.shape[:2]
    M = cv.getRotationMatrix2D((cols / 2, rows / 2), teta_eq, 1)
    dst = cv.warpAffine(img1, M, (cols, rows))
    cv.imwrite(os.path.join(dirname3, f), dst)  # Rotate


# Step 5
# -------------------------------
# Draw Contour



img_folder = os.path.join(os.getcwd(), 'Rotate')
img_files = [f for f in os.listdir(img_folder) if f.endswith('.jpg')]
shape = []
for f in img_files:
    print("Processing Contouring {} ...".format(f))

    img = cv.imread(os.path.join(img_folder, f))
    # (1) Convert to gray, and threshold
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, threshed = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # (2) Morph-op to remove noise
    kernel = np.ones((20, 20), np.uint8)
    morphed = cv.morphologyEx(threshed, cv.MORPH_CLOSE, kernel)

    # (3) Find the max-area contour
    _, cnts, _ = cv.findContours(threshed, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    cnt = sorted(cnts, key=cv.contourArea)[-1]
    # (4) Draw the Extreme Points

    h_img, w_img = img.shape[:2]
    rad = int(w_img / 2) - 42
    ww = int(w_img / 2)
    hh = int(h_img / 2)
    pp = cnt[( ( (cnt[:, 0, 0] - ww)**2 ) + ( (cnt[:, 0, 1] - hh)**2 ) ) < rad**2]
    extLeft = tuple(pp[pp[:, :, 0].argmin()][0])
    extRight = tuple(pp[pp[:, :, 0].argmax()][0])
    extTop = tuple(pp[pp[:, :, 1].argmin()][0])
    extBot = tuple(pp[pp[:, :, 1].argmax()][0])
    cv.circle(img, extLeft, 6, (0, 0, 255), -1)
    cv.circle(img, extTop, 6, (0, 0, 255), -1)
    cv.circle(img, extBot, 6, (0, 0, 255), -1)
    x = extLeft[0]
    y = extTop[1]
    w = extRight[0]
    h = extBot[1]
    dst = img[y: h, x:w]
    height, width = dst.shape[:2]
    shape.append([width, height])
    cv.imwrite(os.path.join(dirname6, f), dst)  #Contour

# # Step 6
# # ---------------------------------------------
# # Scaling
max_h = max(shape, key=lambda x: x[1])[1]
counter = 0
shape1 = []
img_folder = os.path.join(os.getcwd(), 'Contour')
img_files = [f for f in os.listdir(img_folder) if f.endswith('.jpg')]
for f in img_files:
    img = cv.imread(os.path.join(img_folder, f))
    print("Scaling... {}".format(f))
    height, width = img.shape[:2]
    shape[counter][1] = max_h / shape[counter][1]
    w_scaled = int(round(shape[counter][1] * width, 0))
    h_scaled = int(round(shape[counter][1] * height, 0))
    res = cv.resize(img, (w_scaled, h_scaled), interpolation=cv.INTER_CUBIC)
    cv.imwrite(os.path.join(dirname7, f), res)
    height, width = res.shape[:2]
    # print("height = {} , width = {}".format(height, width))
    shape1.append([width, height])
    counter += 1

min_w = min(shape1, key=lambda x: x[0])[0]


# Step 7
# --------------------------------------------
# extracing pixels

ylw_data = []
blue_data = []

result_tests = []
img_names = []

img_folder = os.path.join(os.getcwd(), 'scale')
img_files = [f for f in os.listdir(img_folder) if f.endswith('.jpg')]
for f in img_files:
    print("Real Scaling... ! {}".format(f))

    # making result tests
    ftest = re.sub('.jpg', '', f)
    result_tests.append(float(re.match(r'\d+["."\d+]*', ftest).group()))
    img_names.append(f)

    img = cv.imread(os.path.join(img_folder, f))
    dst = img[0: height, 0:min_w]
    h, w = dst.shape[:2]
    # cv.circle(dst, (int(4*h/15)+5, int(4*h/15)), 40, (0, 255, 255), -1)
    # cv.circle(dst, (int(4*h/15)+5, int(11*h/15)), 40, (0, 255, 255), -1)
    # cv.rectangle(dst, (3*int(4*h/15), int(4*h/15) - 4), (5*int(4*h/15), int(4*h/15)+14), (0, 255, 255), 3)

    # xc1, yc1, rc1, xc2, yc2, rc2 = int(4*h/15)+5, int(4*h/15), int(4*h/31), int(4*h/15)+5, int(11*h/15), int(4*h/31)
    # rectx1, recty1, rectx2, recty2 = 3*int(4*h/15), int(4*h/15) - 4, 5*int(4*h/15), int(4*h/15)+14

    xc1, yc1, rc1, xc2, yc2, rc2 = int(4 * h / 15) + 5, int(4 * h / 15), 25, int(4 * h / 15) + 5, int(
        11 * h / 15), 25
    rectx1, recty1, rectx2, recty2 = int(4 * h / 15) + 200, int(4 * h / 15), int(4 * h / 15) +350, int(4 * h / 15) + 10

    # xc1, yc1, rc1, xc2, yc2, rc2 = int(4 * h / 15) + 5, int(4 * h / 15), 10, int(4 * h / 15) + 5, int(
    #     11 * h / 15), 10
    # rectx1, recty1, rectx2, recty2 = int(4 * h / 15) + 20, int(4 * h / 15), int(4 * h / 15) + 35, int(4 * h / 15) + 10

    ylwpix = []
    for i in range(-rc1, rc1):
        for j in range(-rc1, rc1):
            if i ** 2 + j ** 2 < rc1 ** 2:
                ylwpix.append(255 - img[j + yc1, i + xc1, 0])
                img[j + yc1, i + xc1, :] = (255, 0, 0)

    bluepix = []
    for i in range(-rc2, rc2):
        for j in range(-rc2, rc2):
            if i ** 2 + j ** 2 < rc2 ** 2:
                bluepix.append(255 - img[j + yc2, i + xc2, 2])
                img[j + yc2, i + xc2, :] = (0, 0, 255)


    for i in range(rectx1, rectx2):
        for j in range(recty1, recty2):
            ylwpix.append(255 - img[j, i, 0])
            img[j, i] = (255, 0, 0)

    # making 2D lists for learning
    ylw_data.append(ylwpix)
    blue_data.append(bluepix)

    cv.imwrite(os.path.join(dirname9, f), dst) # circle

# ----------------------------------------------------------------
# Saving datas

# if os.path.exists('test.xlsx'):
#     os.remove('test.xlsx')
# workbook = xlsxwriter.Workbook('test.xlsx')
# worksheet = workbook.add_worksheet()
# worksheet.write(0, 0, 'Blue D')
# worksheet.write(0, 1, 'Yellow D')
# worksheet.write(0, 3, 'Y')
# # print(','.join(str(e) for e in blue_circle[0]))
# for i in range(0, len(blue_data)):
#     worksheet.write(i + 1, 0, ','.join(str(e) for e in blue_data[i]))
#     worksheet.write(i + 1, 1, ','.join(str(e) for e in ylw_data[i]))
#     worksheet.write(i + 1, 3, result_tests[i])
# workbook.close()


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# MACHINE LEARNING STARTS HERE

# Help: result_test, ylw_data, blue_data

from sklearn.neural_network import MLPRegressor


Numberofdatas = len(result_tests)

print("Number of datas: ", Numberofdatas)


from sklearn.preprocessing import StandardScaler

ylwscaler = StandardScaler()
bluescaler = StandardScaler()
resultscaler = StandardScaler()

ylwscaler.fit(ylw_data)
bluescaler.fit(blue_data)
# resultscaler.fit(result_tests)

ylw_data = ylwscaler.transform(ylw_data)
blue_data = bluescaler.transform(blue_data)
# result_tests = resultscaler.transform(result_tests)

# reg = MLPRegressor(hidden_layer_sizes=(10,),  activation='relu', solver='lbfgs',    alpha=0.001,batch_size='auto',
#                learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
#                random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
#                nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
#                epsilon=1e-08)

t1 = process_time()

ylw , blue, ylwtest, bluetest = [], [], [], []

for i in range(Numberofdatas):
    if result_tests[i] > 5:
        ylw.append(ylw_data[i])
        ylwtest.append(result_tests[i])
    elif 5 >= result_tests[i] > 1.4:
        ylw.append(ylw_data[i])
        ylwtest.append(result_tests[i])
        blue.append(blue_data[i])
        bluetest.append(result_tests[i])
    else:
        blue.append(blue_data[i])
        bluetest.append(result_tests[i])


ylwreg = MLPRegressor(hidden_layer_sizes=(len(ylw),) ,solver='lbfgs', alpha=100, activation='identity', max_iter=1000, tol=1e-7, verbose=True, learning_rate='adaptive')
bluereg = MLPRegressor(hidden_layer_sizes=(len(blue),) ,solver='lbfgs', alpha=100, activation='identity', max_iter=1000, tol=1e-7, verbose=True, learning_rate='adaptive')
bothreg = MLPRegressor(hidden_layer_sizes=(len(ylw_data),) ,solver='lbfgs', alpha=100, activation='identity', max_iter=1000, tol=1e-7, verbose=True, learning_rate='adaptive')

bothreg.fit(ylw_data, result_tests)
bothscr = bothreg.score(ylw_data, result_tests)

ylwreg.fit(ylw, ylwtest)
ylwscr = ylwreg.score(ylw, ylwtest)

bluereg.fit(blue, bluetest)
bluescr = bluereg.score(blue, bluetest)


t2 = process_time()

filename1 = 'dinalized_model1.sav'
pickle.dump(bothreg, open(filename1, 'wb'))
filename1 = 'dinalized_model2.sav'
pickle.dump(ylwreg, open(filename1, 'wb'))
filename1 = 'dinalized_model3.sav'
pickle.dump(bluereg, open(filename1, 'wb'))

# Predicting data
bothtrained = bothreg.predict(ylw_data)
ylwtrained = ylwreg.predict(ylw_data)
bluetrained = bluereg.predict(blue_data)
result_trained = []

for i in range(Numberofdatas):
    if bothtrained[i] > 2.5:
        result_trained.append(ylwtrained[i])
    else:
        result_trained.append(bluetrained[i])

for i in range(Numberofdatas):
    print(img_names[i], " : ", result_trained[i] - result_tests[i])

# trainscore = bothreg.score(result_trained,result_tests)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(result_tests, result_trained, s=10, c='r', marker="o", label='Result Prediction')
plt.show()

fig1 = plt.figure()
ax2 = fig1.add_subplot(111)
ax2.scatter(result_tests, bothtrained, s=10, c='r', marker="o", label='both Prediction')
plt.show()

fig1 = plt.figure()
ax2 = fig1.add_subplot(111)
ax2.scatter(result_tests, ylwtrained, s=10, c='r', marker="o", label='ylw Prediction')
plt.show()

fig2 = plt.figure()
ax3 = fig2.add_subplot(111)
ax3.scatter(result_tests, bluetrained, s=10, c='r', marker="o", label='blue Prediction')
plt.show()

# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------
# Predict

dirname1 = 'PCropped'
if os.path.exists(dirname1):
    shutil.rmtree(dirname1)
os.makedirs(dirname1)
dirname2 = 'PColor'
if os.path.exists(dirname2):
    shutil.rmtree(dirname2)
os.makedirs(dirname2)
dirname3 = 'PRotate'
if os.path.exists(dirname3):
    shutil.rmtree(dirname3)
os.makedirs(dirname3)
dirname4 = 'Pdraw'
if os.path.exists(dirname4):
    shutil.rmtree(dirname4)
os.makedirs(dirname4)
dirname5 = 'PBig circle'
if os.path.exists(dirname5):
    shutil.rmtree(dirname5)
os.makedirs(dirname5)
dirname6 = 'PContour'
if os.path.exists(dirname6):
    shutil.rmtree(dirname6)
os.makedirs(dirname6)
dirname7 = 'Pscale'
if os.path.exists(dirname7):
    shutil.rmtree(dirname7)
os.makedirs(dirname7)
dirname9 = 'Pcircle'
if os.path.exists(dirname9):
    shutil.rmtree(dirname9)
os.makedirs(dirname9)

# Step 1
# -------------------------------
img_folder = os.path.join(os.getcwd(), dirname_predict)
img_files = [f for f in os.listdir(img_folder) if f.endswith('.jpg')]
for f in img_files:
    cimg = cv.imread(os.path.join(img_folder, f))
    gray = cv.cvtColor(cimg, cv.COLOR_BGR2GRAY)
    print("processing.... {}".format(f))
    circles = cv.HoughCircles(gray,cv.HOUGH_GRADIENT,10,1800, maxRadius=800, minRadius=50)
    circles = np.uint16(np.around(circles))
    x, y, r = 0, 0, 0
    for i in circles[0,:]:
        x, y, r = i[0], i[1], i[2]
    dst = cimg[y-r:y + r, x-r:x + r]
    cv.imwrite(os.path.join(dirname1, f), dst)

# Step 2
# ---------------------------------------------
# Find the extreme points and Crop the image
# Big one part

img_folder = os.path.join(os.getcwd(), 'PCropped')
img_files = [f for f in os.listdir(img_folder) if f.endswith('.jpg')]

for f in img_files:
    print("Processing Contouring {} ...".format(f))
    img = cv.imread(os.path.join(img_folder, f))
    # (1) Convert to gray, and threshold
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, threshed = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # (2) Morph-op to remove noise
    kernel = np.ones((20, 20), np.uint8)
    morphed = cv.morphologyEx(threshed, cv.MORPH_CLOSE, kernel)

    # (3) Find the max-area contour
    _, cnts, _ = cv.findContours(threshed, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    cnt = sorted(cnts, key=cv.contourArea)[-1]

    # (4) Draw the Extreme Points

    h_img, w_img = img.shape[:2]

    extLeft = tuple(cnt[cnt[:, :, 0].argmin()][0])
    extRight = tuple(cnt[cnt[:, :, 0].argmax()][0])
    extTop = tuple(cnt[cnt[:, :, 1].argmin()][0])
    extBot = tuple(cnt[cnt[:, :, 1].argmax()][0])

    x = extLeft[0]
    w = extRight[0]
    y = extTop[1]
    h = extBot[1]
    dst = img[y: h, x:w-40]
    height, width = dst.shape[:2]

    cv.imwrite(os.path.join(dirname5, f), dst) # Big one circle

# Step 3
# -------------------------------
img_folder = os.path.join(os.getcwd(), 'PBig circle')
img_files = [f for f in os.listdir(img_folder) if f.endswith('.jpg')]

for f in img_files:
    print("STEP 2 - Cropping  {} ...".format(f))
    img = cv.imread(os.path.join(img_folder, f))

    x, y, w, h = 350, 200, 450, 330
    dst = img[y:y + h, x:x + w]

    cv.imwrite(os.path.join(dirname2, f), dst) # Color -> shaft line

# # Step 4
# # -----------------------------------------------------
# # Rotate the images from Step 1

img_folder = os.path.join(os.getcwd(), 'PColor')
img_files = [f for f in os.listdir(img_folder) if f.endswith('.jpg')]
img_folder1 = os.path.join(os.getcwd(), 'PBig circle')
img_files1 = [f for f in os.listdir(img_folder1) if f.endswith('.jpg')]

for f in img_files:
    img = cv.imread(os.path.join(img_folder, f))
    # (1) Convert to gray, and threshold
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, threshed = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # (2) Morph-op to remove noise
    kernel = np.ones((20, 20), np.uint8)
    morphed = cv.morphologyEx(threshed, cv.MORPH_CLOSE, kernel)

    # (3) Find the max-area contour
    _, cnts, _ = cv.findContours(morphed, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    cnts = sorted(cnts, key=cv.contourArea)

    h_img, w_img = img.shape[:2]
    def give_teta(cnt, conter=0):
        pixelpoints = cnt[np.logical_and(20 < cnt[:, 0, 0], cnt[:, 0, 0] < w_img - 45)]
        pixelpoints = pixelpoints[np.logical_and(20 < pixelpoints[:, 0, 1], pixelpoints[:, 0, 1] < h_img - 10)]
        x = pixelpoints[:, :, 0].flatten()
        y = pixelpoints[:, :, 1].flatten()
        cv.drawContours(img, [pixelpoints], 0, (255, 0, 0), 2)
        cv.imwrite(os.path.join(dirname4, f), img)
        slope, _, _, _, _ = stats.linregress(x, y)

        teta = np.arctan(slope)*180/np.pi
        return teta

    cnt = cnts[-1]
    teta1 = give_teta(cnt,1)
    cnt = cnts[-2]
    teta2 = give_teta(cnt)
    # cnt = cnts[-3]
    # teta3 = give_teta(cnt)
    # print(teta1, teta2, teta3)
    # teta4 = give_teta(cnts[-4])
    teta_eq = ((teta1 + teta2)/2)

    print("Rotating {} ...".format(f))
    # print(teta1, " + ", teta2, " + ", teta3, " = ", teta_eq)
    rows, cols = img.shape[:2]

    img1 = cv.imread(os.path.join(img_folder1, f))
    rows, cols = img1.shape[:2]
    M = cv.getRotationMatrix2D((cols / 2, rows / 2), teta_eq, 1)
    dst = cv.warpAffine(img1, M, (cols, rows))
    cv.imwrite(os.path.join(dirname3, f), dst)  # Rotate


# Step 5
# -------------------------------
# Draw Contour



img_folder = os.path.join(os.getcwd(), 'PRotate')
img_files = [f for f in os.listdir(img_folder) if f.endswith('.jpg')]
shape = []
for f in img_files:
    print("Processing Contouring {} ...".format(f))

    img = cv.imread(os.path.join(img_folder, f))
    # (1) Convert to gray, and threshold
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, threshed = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # (2) Morph-op to remove noise
    kernel = np.ones((20, 20), np.uint8)
    morphed = cv.morphologyEx(threshed, cv.MORPH_CLOSE, kernel)

    # (3) Find the max-area contour
    _, cnts, _ = cv.findContours(threshed, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    cnt = sorted(cnts, key=cv.contourArea)[-1]
    # (4) Draw the Extreme Points

    h_img, w_img = img.shape[:2]
    rad = int(w_img / 2) - 42
    ww = int(w_img / 2)
    hh = int(h_img / 2)
    pp = cnt[( ( (cnt[:, 0, 0] - ww)**2 ) + ( (cnt[:, 0, 1] - hh)**2 ) ) < rad**2]
    extLeft = tuple(pp[pp[:, :, 0].argmin()][0])
    extRight = tuple(pp[pp[:, :, 0].argmax()][0])
    extTop = tuple(pp[pp[:, :, 1].argmin()][0])
    extBot = tuple(pp[pp[:, :, 1].argmax()][0])
    cv.circle(img, extLeft, 6, (0, 0, 255), -1)
    cv.circle(img, extTop, 6, (0, 0, 255), -1)
    cv.circle(img, extBot, 6, (0, 0, 255), -1)
    x = extLeft[0]
    y = extTop[1]
    w = extRight[0]
    h = extBot[1]
    dst = img[y: h, x:w]
    height, width = dst.shape[:2]
    shape.append([width, height])
    cv.imwrite(os.path.join(dirname6, f), dst)  #Contooor

# # Step 6
# # ---------------------------------------------
# # Scaling
max_h = max(shape, key=lambda x: x[1])[1]
counter = 0
shape1 = []
img_folder = os.path.join(os.getcwd(), 'PContour')
img_files = [f for f in os.listdir(img_folder) if f.endswith('.jpg')]
for f in img_files:
    img = cv.imread(os.path.join(img_folder, f))
    print("Scaling... {}".format(f))
    height, width = img.shape[:2]
    shape[counter][1] = max_h / shape[counter][1]
    w_scaled = int(round(shape[counter][1] * width, 0))
    h_scaled = int(round(shape[counter][1] * height, 0))
    res = cv.resize(img, (w_scaled, h_scaled), interpolation=cv.INTER_CUBIC)
    cv.imwrite(os.path.join(dirname7, f), res)
    height, width = res.shape[:2]
    # print("height = {} , width = {}".format(height, width))
    shape1.append([width, height])
    counter += 1

min_w = min(shape1, key=lambda x: x[0])[0]


# Step 7
# --------------------------------------------
# extracing pixels

ylw_data = []
blue_data = []

# result_tests = []
img_names = []

img_folder = os.path.join(os.getcwd(), 'Pscale')
img_files = [f for f in os.listdir(img_folder) if f.endswith('.jpg')]
for f in img_files:
    print("Real Scaling... ! {}".format(f))

    # making result tests
    # ftest = re.sub('.jpg', '', f)
    # result_tests.append(float(re.match(r'\d+["."\d+]*', ftest).group()))
    img_names.append(f)

    img = cv.imread(os.path.join(img_folder, f))
    dst = img[0: height, 0:min_w]
    h, w = dst.shape[:2]
    # cv.circle(dst, (int(4*h/15)+5, int(4*h/15)), 40, (0, 255, 255), -1)
    # cv.circle(dst, (int(4*h/15)+5, int(11*h/15)), 40, (0, 255, 255), -1)
    # cv.rectangle(dst, (3*int(4*h/15), int(4*h/15) - 4), (5*int(4*h/15), int(4*h/15)+14), (0, 255, 255), 3)

    # xc1, yc1, rc1, xc2, yc2, rc2 = int(4*h/15)+5, int(4*h/15), int(4*h/31), int(4*h/15)+5, int(11*h/15), int(4*h/31)
    # rectx1, recty1, rectx2, recty2 = 3*int(4*h/15), int(4*h/15) - 4, 5*int(4*h/15), int(4*h/15)+14

    xc1, yc1, rc1, xc2, yc2, rc2 = int(4 * h / 15) + 5, int(4 * h / 15), 25, int(4 * h / 15) + 5, int(
        11 * h / 15), 25
    rectx1, recty1, rectx2, recty2 = int(4 * h / 15) + 200, int(4 * h / 15), int(4 * h / 15) +350, int(4 * h / 15) + 10

    # xc1, yc1, rc1, xc2, yc2, rc2 = int(4 * h / 15) + 5, int(4 * h / 15), 10, int(4 * h / 15) + 5, int(
    #     11 * h / 15), 10
    # rectx1, recty1, rectx2, recty2 = int(4 * h / 15) + 20, int(4 * h / 15), int(4 * h / 15) + 35, int(4 * h / 15) + 10

    ylwpix = []
    for i in range(-rc1, rc1):
        for j in range(-rc1, rc1):
            if i ** 2 + j ** 2 < rc1 ** 2:
                ylwpix.append(255 - img[j + yc1, i + xc1, 0])
                img[j + yc1, i + xc1, :] = (255, 0, 0)

    bluepix = []
    for i in range(-rc2, rc2):
        for j in range(-rc2, rc2):
            if i ** 2 + j ** 2 < rc2 ** 2:
                bluepix.append(255 - img[j + yc2, i + xc2, 2])
                img[j + yc2, i + xc2, :] = (0, 0, 255)


    for i in range(rectx1, rectx2):
        for j in range(recty1, recty2):
            ylwpix.append(255 - img[j, i, 0])
            img[j, i] = (255, 0, 0)

    # making 2D lists for learning
    ylw_data.append(ylwpix)
    blue_data.append(bluepix)

    cv.imwrite(os.path.join(dirname9, f), dst) # circle


# --------------------------------------------------
# --------------------------------------------------
# predict data by learning



from sklearn.preprocessing import StandardScaler

ylwscaler = StandardScaler()
bluescaler = StandardScaler()

ylwscaler.fit(ylw_data)
bluescaler.fit(blue_data)

ylw_data = ylwscaler.transform(ylw_data)
blue_data = bluescaler.transform(blue_data)

bothtrained = bothreg.predict(ylw_data)
ylwtrained = ylwreg.predict(ylw_data)
bluetrained = bluereg.predict(blue_data)
result_trained = []

Numberofdatas = len(bothtrained)

for i in range(Numberofdatas):
    if bothtrained[i] > 2.5:
        result_trained.append(ylwtrained[i])
    else:
        result_trained.append(bluetrained[i])

workbook = xlsxwriter.Workbook('complete this ans.xlsx')
worksheet = workbook.add_worksheet()

for i in range(Numberofdatas):
    print(img_names[i], " : ", result_trained[i]) # - result_tests[i])
    worksheet.write(i + 1, 1, result_trained[i])

# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.scatter(result_tests, result_trained, s=10, c='r', marker="o", label='Result Prediction')
# plt.show()
#
# fig1 = plt.figure()
# ax2 = fig1.add_subplot(111)
# ax2.scatter(result_tests, bothtrained, s=10, c='r', marker="o", label='both Prediction')
# plt.show()
#
# fig1 = plt.figure()
# ax2 = fig1.add_subplot(111)
# ax2.scatter(result_tests, ylwtrained, s=10, c='r', marker="o", label='ylw Prediction')
# plt.show()
#
# fig2 = plt.figure()
# ax3 = fig2.add_subplot(111)
# ax3.scatter(result_tests, bluetrained, s=10, c='r', marker="o", label='blue Prediction')
# plt.show()

bothscr1 = bothreg.score(ylw_data, bothtrained)
ylwscr1 = ylwreg.score(ylw_data, ylwtrained)
bluescr1 = bluereg.score(blue_data, bluetrained)

print('Learning time: ', t2 - t1, 's')

print('ylw fit score: ', ylwscr)
print('blue fit score: ', bluescr)
print('both fit score: ', bothscr)

print('ylw predict score: ', ylwscr1)
print('blue predict score: ', bluescr1)
print('both predict score: ', bothscr1)

# --------------------------------------------------
# Saving Pixels

# if os.path.exists('test1.xlsx'):
#     os.remove('test1.xlsx')
# workbook = xlsxwriter.Workbook('test1.xlsx')
# worksheet = workbook.add_worksheet()
# worksheet.write(0, 0, 'Blue D')
# worksheet.write(0, 1, 'Yellow D')
# worksheet.write(0, 3, 'Y')
# # print(','.join(str(e) for e in blue_circle[0]))
# for i in range(0, len(blue_data)):
#     worksheet.write(i + 1, 0, ','.join(str(e) for e in blue_data[i]))
#     worksheet.write(i + 1, 1, ','.join(str(e) for e in ylw_data[i]))
#     worksheet.write(i + 1, 3, result_trained[i])
# workbook.close()

workbook.close()
