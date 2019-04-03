"""
    Author : YangBo
    Time : 2019-04-03 23:01
    function:
"""
import cv2
import glob
files = glob.glob('./*g')
print(files)
for file in files:
    image=cv2.imread(file,0)
    # cv2.imshow('picture', image)
    cv2.waitKey(1000)
    print(image)
    new_image = cv2.resize(image,(64,64))
    cv2.imshow('picture',new_image)
    if image is None:
        print('图片为空')
    else:
        print('图片不为空')