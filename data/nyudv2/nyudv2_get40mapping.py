import scipy.io
import numpy as np

#https://github.com/ankurhanda/nyuv2-meta-data

f_40_ = scipy.io.loadmat('nyudv2_mapping40.mat')
print(f_40_.keys())

class40_names_ = np.squeeze(np.array(f_40_.get('className')))
for i in range(class40_names_.shape[0]):
  class40_names_[i] = str(class40_names_[i][0])
print(class40_names_.shape)

all_classes_ = np.squeeze(np.array(f_40_.get('allClassName')))
for i in range(all_classes_.shape[0]):
  all_classes_[i] = str(all_classes_[i][0])
print(all_classes_.shape)

map_classes_40_ = np.squeeze(np.array(f_40_.get('mapClass')))
print(map_classes_40_.shape)

print("all_classes_ = {")
for i in range(all_classes_.shape[0]):
  print("{0} : {1},".format(all_classes_[i], i))
print("}")

print("classes_40_ = {")
print("\"void\" : 0,")
for i in range(class40_names_.shape[0]):
  print("\"{0}\" : {1},".format(class40_names_[i], i+1))
print("}")

print("map_classes_40_ = [")
for i in range(map_classes_40_.shape[0]):
  print("{0},".format(map_classes_40_[i]), end="")
print("]")
