import os
from PIL import Image
import numpy as np

class ADE20KLoader():

    BASE_DIR = 'ADEChallengeData2016'
    
    def __init__(self, root=os.getcwd(),
                 split='train', **kwargs):
        root = os.path.join(root, self.BASE_DIR)
        assert os.path.exists(root), "Error with the dataset path; make sure the data is found in the root dir/"
        self.images, self.masks = self._get_ade20k_pairs(root, split)
        assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: \
                " + root + "\n"))
        self.idx_fid = np.array([path.split('/')[-1].replace('.jpg', '') for path in self.images])  # mapping image index to its file id to get the class
        self.img_label = self._get_classes(self.BASE_DIR+'/sceneCategories.txt')   # getting the categorical label of each image ex: img_label[fid]
        self.labels = np.array([self.img_label[fid] for fid in self.idx_fid])    # getting label array ex: labels[0]

    def __getitem__(self, index, return_msk = False):
        img = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.masks[index]) if return_msk else None
        return (img, mask) if return_msk else img

    def __len__(self):
        return len(self.images)

    def _get_classes(self, filename):
        img_label = {}
        with open(filename) as f:
            lines = f.read().splitlines()
            splitted_lines = np.array([line.split() for line in lines])
            for fid, label in zip(splitted_lines[:,0], splitted_lines[:,1]):
                img_label[fid] = label
        return img_label

    def _get_ade20k_pairs(self, folder, mode='train'):
	    img_paths = []
	    mask_paths = []
	    if mode == 'train':
	        img_folder = os.path.join(folder, 'images/training')
	        mask_folder = os.path.join(folder, 'annotations/training')
	    else:
	        img_folder = os.path.join(folder, 'images/validation')
	        mask_folder = os.path.join(folder, 'annotations/validation')
	    for filename in os.listdir(img_folder):
	        basename, _ = os.path.splitext(filename)
	        if filename.endswith(".jpg"):
	            imgpath = os.path.join(img_folder, filename)
	            maskname = basename + '.png'
	            maskpath = os.path.join(mask_folder, maskname)
	            if os.path.isfile(maskpath):
	                img_paths.append(imgpath)
	                mask_paths.append(maskpath)
	            else:
	                print('cannot find the mask:', maskpath)
	    return np.array(img_paths), np.array(mask_paths)
