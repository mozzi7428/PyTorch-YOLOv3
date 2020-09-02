import shutil
import glob
import json
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--kth_images", type=int, default=10, help="Use 1/k of total images to train and validate")
parser.add_argument("--train_ratio", type=float, default=0.8, help="train:val = ratio : 1-ratio")
opt = parser.parse_args()
print(opt)

print(opt.kth_images)
print(opt.train_ratio)

if not os.path.exists('./images'):
    os.mkdir('./images')
if not os.path.exists('./labels'):
    os.mkdir('./labels')


## Rename move images ========================
fnames = glob.glob('./pedestrian/image/*/*.jpg')

for fname in fnames:
    video, frame = fname.split('/')[-2:]
    shutil.move(fname, './images/%s__%s'%(video, frame))


## Retrieve annotation information ============
print('Retrieving annotation information')

# collect list of images
image_fnames = glob.glob('./images/*.jpg')

image_dict = {}
for image_fname in image_fnames:
	video, frame = image_fname.split('/')[-1].split('__')
	frame = frame.split('_')[-1][:-len('.jpg')]
	image_dict[(video, frame)] = []
	

# collect bounding box information for each images
anno_fnames = glob.glob('pedestrian/label/*.json')
r, c = 1080., 1920.,

for i, anno_fname in enumerate(anno_fnames, start=1):
	print('%d/%d'%(i, len(anno_fnames)))
	video = anno_fname.split('/')[-1][:-len('-labels.json')]
	annos = json.load(open(anno_fname))
	
	for anno in annos:
		frame = str(anno['frame'])
		x, y = anno['coordinates'][0]
		x_, y_ = anno['coordinates'][1]

		if np.all([x, y, x_, y_]) and anno['label']=='P':
			label_idx = 0
			x_center = (x+x_)/(2*c)
			y_center = (y+y_)/(2*r)
			height = abs(y-y_)/r
			width = abs(y-y_)*0.41/c

			image_dict[(video, frame)].append([label_idx, x_center, y_center, width, height])

for key in image_dict:
	video, frame = key
	if len(image_dict[key])<1:
		continue
	else:
		label_fname = './labels/%s__frames_%s.txt'%(video, frame)
		label = open(label_fname, 'w')
		for anno in image_dict[key]:
			label_idx, x_center, y_center, width, height = anno
			label.write('%d %f %f %f %f\n'%(label_idx, x_center, y_center, width, height))
		label.close()


# Split train and validation set ================
label_fnames = glob.glob('./labels/*.txt')
label_fnames = np.sort(label_fnames)

k = opt.kth_images			# Use 1/k of total images
ratio = opt.train_ratio		# train:val = ratio : 1-ratio

label_fnames = [label_fnames[i*k] for i in range(len(label_fnames)//k)]

train_images = open('./train.txt', 'w')
for label_fname in label_fnames[:int(ratio*len(label_fnames))]:
    train_images.write('data/custom/images/%s\n'%(label_fname.split('/')[-1].replace('txt', 'jpg')))
train_images.close()

valid_images = open('./valid.txt', 'w')
for label_fname in label_fnames[int(ratio*len(label_fnames)):]:
    valid_images.write('data/custom/images/%s\n'%(label_fname.split('/')[-1].replace('txt','jpg')))
valid_images.close()

