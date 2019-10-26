from imgaug import augmenters as iaa
import imgaug as ia
from xml.etree import ElementTree
import os
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import matplotlib.pyplot as plt
import cv2



ia.seed(42)
os.chdir(os.path.abspath('C:/Users/Vidur/Pictures/Supermarket_Dataset/Milk'))


IMG_SIZE=300
seq = iaa.Sequential([
    iaa.Resize({"height": IMG_SIZE, "width": IMG_SIZE})
])


img_names = ['IMG_20191003_145533']



def extract_boxes(filename):
	# load and parse the file
	tree = ElementTree.parse(filename)
	# get the root of the document
	root = tree.getroot()
	# extract each bounding box
	boxes = list()
	for box in root.findall('.//bndbox'):
		xmin = int(box.find('xmin').text)
		ymin = int(box.find('ymin').text)
		xmax = int(box.find('xmax').text)
		ymax = int(box.find('ymax').text)
		coors = [xmin, ymin, xmax, ymax]
		boxes.append(coors)
	# extract image dimensions
	width = int(root.find('.//size/width').text)
	height = int(root.find('.//size/height').text)
	return boxes, width, height




def map_to_BoundingBoxes(boxes, img_h, img_w):
	print(image.shape)
	h,w,c = image.shape
	bounding_boxes=[]
	for xmin, ymin, xmax, ymax in boxes:
		bounding_boxes.append(BoundingBox(x1=xmin, y1=ymin, x2=xmax, y2=ymax))
	bounding_boxes = BoundingBoxesOnImage(bounding_boxes, shape=(img_h,img_w))
	return bounding_boxes


image = plt.imread(img_names[0]+'.jpg')
boxes, img_w, img_h = extract_boxes(img_names[0]+'.xml')
bbs = map_to_BoundingBoxes(boxes,img_h, img_w)
image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
image_before = bbs.draw_on_image(image, size=2)
image_after = bbs_aug.draw_on_image(image_aug, size=5, color=[0, 0, 255])
plt.imshow(image_before)
plt.show()
plt.imshow(image_after)
plt.show()