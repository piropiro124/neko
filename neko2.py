from PIL import Image
import os, glob
import numpy as np
import random, math

# 画像が保存されているディレクトリのパス
root_dir = "/Users/ayano/Downloads/neko_collector-main/neko_collector/Face"
# 画像が保存されているフォルダ名
categories = ["sinkeisitu","otonasi","natukko", "yancha"]

X = [] # 画像データ
Y = [] # ラベルデータ

# フォルダごとに分けられたファイルを収集
#（categoriesのidxと、画像のファイルパスが紐づいたリストを生成）
allfiles = []
for idx, cat in enumerate(categories):
    image_dir = root_dir + "/" + cat + "_face"
    files = glob.glob(image_dir + "/*")
    for f in files:
        allfiles.append((idx, f))

for cat, fname in allfiles:
    img = Image.open(fname)
    img = img.convert("RGB")
    img = img.resize((150, 150))
    data = np.asarray(img)
    X.append(data)
    Y.append(cat)

x = np.array(X)
y = np.array(Y)

np.save("/Users/ayano/Downloads/neko_collector-main/neko_collector/test/neko_data_test_X_150.npy", x)
np.save("/Users/ayano/Downloads/neko_collector-main/neko_collector/test/neko_data_test_Y_150.npy", y)


