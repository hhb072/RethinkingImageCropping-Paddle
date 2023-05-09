import os
import numpy as np
import shutil

newcpcroot = '/data1/CPCDataset_spit/'
if not os.path.exists(newcpcroot):
    os.mkdir(newcpcroot)
newsub1 = '/data1/CPCDataset_spit/images/train'
if not os.path.exists(newsub1):
    os.mkdir(newsub1)
newsub2 = '/data1/CPCDataset_spit/images/val'
if not os.path.exists(newsub2):
    os.mkdir(newsub2)
newsub3 = '/data1/CPCDataset_spit/CollectedAnnotationsRaw/train'
if not os.path.exists(newsub3):
    os.mkdir(newsub3)
newsub4 = '/data1/CPCDataset_spit/CollectedAnnotationsRaw/val'
if not os.path.exists(newsub4):
    os.mkdir(newsub4)

oldcpcroot = '/data1/CPCDataset/'

imgdir = os.path.join(oldcpcroot, 'images/')
annodir = os.path.join(oldcpcroot, 'CollectedAnnotationsRaw/')
imglist = os.listdir(imgdir)
imgpath = []
annopath = []
for image in imglist:
    annofile = os.path.join(annodir, image + '.txt')

    if os.path.exists(annofile):
        imgpath.append(os.path.join(imgdir, image))
        annopath.append(os.path.join(annodir, image + '.txt'))


ids = list(range(len(imgpath)))
np.random.shuffle(ids)

trainimgpath = []
trainannopath = []

for id in ids[:9797]:
    trainimgpath.append(imgpath[id])
    trainannopath.append(annopath[id])

valimgpath = []
valannopath = []

for id in ids[9797:]:
    valimgpath.append(imgpath[id])
    valannopath.append(annopath[id])

for image, anno in zip(trainimgpath, trainannopath):

    newimage = os.path.join(newcpcroot, 'images/train', image.split('/')[-1])

    shutil.copy(image, newimage)

    newanno = os.path.join(newcpcroot, 'CollectedAnnotationsRaw/train', anno.split('/')[-1])
    shutil.copy(anno, newanno)


for image, anno in zip(valimgpath, valannopath):

    newimage = os.path.join(newcpcroot, 'images/val', image.split('/')[-1])
    shutil.copy(image, newimage)

    newanno = os.path.join(newcpcroot, 'CollectedAnnotationsRaw/val', anno.split('/')[-1])
    shutil.copy(anno, newanno)