import os
import shutil

# os.makedirs('/data1/GAIC_split/images/train/')
# os.makedirs('/data1/GAIC_split/images/test/')
# os.makedirs('/data1/GAIC_split/images/val/')
#
# os.makedirs('/data1/GAIC_split/annotations/train/')
# os.makedirs('/data1/GAIC_split/annotations/test/')
# os.makedirs('/data1/GAIC_split/annotations/val/')


extrain = '/data1/GAIC/42_9528_train.txt'
exval = '/data1/GAIC/42_9528_val.txt'

with open(extrain, "r") as f:
    temptrainlist = f.readlines()
with open(exval, "r") as f:
    tempvallist = f.readlines()

real_trainnum = int(len(temptrainlist[0]) / 10)
real_valnum = int(len(tempvallist[0]) / 10)

trainlist = []
for i in range(real_trainnum):
    trainlist.append(temptrainlist[0][i * 10:(i + 1) * 10])
vallist = []
for i in range(real_valnum):
    vallist.append(tempvallist[0][i * 10:(i + 1) * 10])

print(trainlist)
print(vallist)

oldroot = '/data1/GAIC/'
newroot = '/data1/GAIC_split/'

for trainimgname in trainlist:
    oldfile = os.path.join(oldroot, 'images/train/', trainimgname)
    newfile = os.path.join(newroot, 'images/train/', trainimgname)
    shutil.copy(oldfile, newfile)

    oldfileanno = os.path.join(oldroot, 'annotations/', trainimgname[:-3] + 'txt')
    newfileanno = os.path.join(newroot, 'annotations/train/', trainimgname[:-3] + 'txt')
    shutil.copy(oldfileanno, newfileanno)


for valimgname in vallist:
    oldfile = os.path.join(oldroot, 'images/train/', valimgname)
    newfile = os.path.join(newroot, 'images/val/', valimgname)
    shutil.copy(oldfile, newfile)

    oldfileanno = os.path.join(oldroot, 'annotations/', valimgname[:-3] + 'txt')
    newfileanno = os.path.join(newroot, 'annotations/val/', valimgname[:-3] + 'txt')
    shutil.copy(oldfileanno, newfileanno)

testlist = os.listdir('/data1/GAIC/images/test/')

for testimgname in testlist:
    oldfile = os.path.join(oldroot, 'images/test/', testimgname)
    newfile = os.path.join(newroot, 'images/test/', testimgname)
    shutil.copy(oldfile, newfile)

    oldfileanno = os.path.join(oldroot, 'annotations/', testimgname[:-3] + 'txt')
    newfileanno = os.path.join(newroot, 'annotations/test/', testimgname[:-3] + 'txt')
    shutil.copy(oldfileanno, newfileanno)