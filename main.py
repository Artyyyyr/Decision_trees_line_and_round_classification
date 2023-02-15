from PIL import Image
import os

import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn import tree

in_link_lin = []
in_link_ron = []

for file in os.listdir(r"img_lin"):
    file = "img_lin\\" + file
    in_link_lin.append(file)
for file in os.listdir(r"img_ron"):
    file = "img_ron\\" + file
    in_link_ron.append(file)

in_imgs_lin = []
in_imgs_ron = []

for i in in_link_lin:
    in_imgs_lin.append(Image.open(i))
for i in in_link_ron:
    in_imgs_ron.append(Image.open(i))


im_gis_lin = []
for i in range(len(in_link_lin)):
    im_gis_lin.append(in_imgs_lin[i].histogram())

im_gis_ron = []
for i in range(len(in_link_ron)):
    im_gis_ron.append(in_imgs_ron[i].histogram())

Just_tree = tree.DecisionTreeClassifier()

data_y = []
data_x = []

for i in range(len(in_imgs_lin)):
    data_y.append(1)
    data_x.append(im_gis_lin[i])

for i in range(len(in_imgs_ron)):
    data_y.append(0)
    data_x.append(im_gis_ron[i])

Just_tree.fit(data_x, data_y)

#  Test with new data
in_link_new = []
print("RESULT FROM FILES FOR TESTING")
print("JUST TREE")
for file in os.listdir(r"new_img"):
    file = "new_img\\" + file
    in_link_new.append(file)

in_imgs_new = []
for i in in_link_new:
    in_imgs_new.append(Image.open(i))

im_gis_new = []
for i in range(len(in_link_new)):
    im_gis_new.append(in_imgs_new[i].histogram())
res = Just_tree.predict(im_gis_new)
for i in range(len(in_imgs_new)):
    if res[i] == 0:
        print("file: " + str(in_link_new[i]) + " is opened. Result: circle")
    else:
        print("file: " + str(in_link_new[i]) + " is opened. Result: line")
#  Forest

Forest = RandomForestClassifier(n_estimators=1000)

Forest.fit(data_x, data_y)
print("FOREST")
for i in range(len(in_imgs_new)):
    print("file: " + str(in_link_new[i]) + " is opened. Result: circle - " +
          str(round(Forest.predict_proba([im_gis_new[i]])[0][0] * 1000)/10) + "%, line - " +
          str(round(Forest.predict_proba([im_gis_new[i]])[0][1] * 1000)/10) + "%")

#  AdaBoost
Ada = AdaBoostClassifier()
Ada.fit(data_x, data_y)

print("ADABOOST")
for i in range(len(in_imgs_new)):
    print("file: " + str(in_link_new[i]) + " is opened. Result: circle - " +
          str(round(Ada.predict_proba([im_gis_new[i]])[0][0] * 1000) / 10) + "%, line - " +
          str(round(Ada.predict_proba([im_gis_new[i]])[0][1] * 1000) / 10) + "%")

Grad = GradientBoostingClassifier()
Grad.fit(data_x, data_y)

#  GRADIENTBOOSTINGCLASSIFIER
print("GRADIENTBOOSTINGCLASSIFIER")
for i in range(len(in_imgs_new)):
    print("file: " + str(in_link_new[i]) + " is opened. Result: circle - " +
          str(round(Grad.predict_proba([im_gis_new[i]])[0][0] * 1000)/10) +
          "%, line - " + str(round(Grad.predict_proba([im_gis_new[i]])[0][1] * 1000)/10) + "%")

#  XGBOOSTING
print("XGBOOSTING")
XGB = xgb.XGBClassifier(objective="binary:logistic")
XGB.fit(data_x, data_y)

for i in range(len(in_imgs_new)):
    print("file: " + str(in_link_new[i]) + " is opened. Result: circle - " +
          str(round(XGB.predict_proba([im_gis_new[i]])[0][0] * 1000)/10) +
          "% , line - " + str(round(XGB.predict_proba([im_gis_new[i]])[0][1] * 1000)/10) + "%")

plt.figure("JUST TREE")
tree.plot_tree(Just_tree)
plt.show()
