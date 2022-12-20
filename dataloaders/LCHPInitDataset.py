import os
import sys

sys.path.append(os.path.abspath(os.getcwd()))


from torch.utils.data import Dataset
import torchvision.transforms as transforms

import torch
import imgaug as ia
import imgaug.augmenters as iaa
import cv2



def get_image_ids(file_path):
    """
    image_ids.txt has the structure

    <path>
    path/to/image1.jpg
    path/to/image2.jpg
    path/to/image3.jpg
    ...
    """
    image_ids = []
    with open(file_path) as f:
        for line in f.readlines():
            image_ids.append(line.strip('\n'))
    return image_ids


def get_class_labels(file_path):
    """
    image_ids.txt has the structure

    <path>,<integer_class_label>
    path/to/image1.jpg,0
    path/to/image2.jpg,1
    path/to/image3.jpg,1
    ...
    """
    class_labels = {}
    with open(file_path) as f:
        for line in f.readlines():
            image_id, class_label_string = line.strip('\n').split(',')
            class_labels[image_id] = int(class_label_string)
    return class_labels


def get_bounding_boxes(file_path, y0x0y1x1=False):
    """
    localization.txt (for bounding box) has the structure

    <path>,<x0>,<y0>,<x1>,<y1>
    path/to/image1.jpg,156,163,318,230
    path/to/image1.jpg,23,12,101,259
    path/to/image2.jpg,143,142,394,248
    path/to/image3.jpg,28,94,485,303
    ...

    One image may contain multiple boxes (multiple boxes for the same path).
    """
    boxes = {}
    with open(file_path) as f:
        for line in f.readlines():
            image_id, x0s, y0s, x1s, y1s = line.strip('\n').split(',')
            a, b, c, d = int(x0s), int(y0s), int(x1s), int(y1s)
            if y0x0y1x1:
                a, b, c, d = int(y0s), int(x0s), int(y1s), int(x1s)
            if image_id in boxes:
                boxes[image_id].append((a, b, c, d))
            else:
                boxes[image_id] = [(a, b, c, d)]
    return boxes



class LCHPInitDataset(Dataset):
    """
    # Description:
        Dataset for retrieving CUB-200-2011 images and labels

    # Member Functions:
        __init__(self, phase, resize):  initializes a dataset
            phase:                      a string in ['train', 'val', 'test']
            resize:                     output shape/size of an image

        __getitem__(self, item):        returns an image
            item:                       the idex of image in the whole dataset

        __len__(self):                  returns the length of dataset
    """

    def __init__(self, dataset='CUB', meta_path='metadata/', dataset_path='/root/data/', phase='train', resize=(448,448), y0x0y1x1=True):
        assert phase in ['train', 'val']
        self.phase = phase
        self.resize = resize

        self.image_ids = get_image_ids(os.path.join(meta_path, dataset, phase, 'image_ids.txt'))
        self.class_labels = get_class_labels(os.path.join(meta_path, dataset, phase, 'class_labels.txt'))
        self.bounding_boxes = get_bounding_boxes(os.path.join(meta_path, dataset, phase, 'localization.txt'), y0x0y1x1=y0x0y1x1)

        self.DATAPATH = os.path.join(dataset_path, dataset)
        self.METAPATH = meta_path
        self.iaaSeq = self.getiaaAug(phase=phase)
        self.num_classes = len(set(self.class_labels.values()))

        # print(len(self.image_ids), len(self.class_labels), len(self.bounding_boxes))
    

    def __getitem__(self, i):
        image_id = self.image_ids[i]
        label = self.class_labels[image_id]
        locs = self.bounding_boxes[image_id]
        # print(locs)

        # read image
        image = cv2.imread(os.path.join(self.DATAPATH, image_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height,width,_ = image.shape

        # transform bounding boxes
        iaa_bboxes = []
        for loc in locs:
            bbox = ia.BoundingBox(y1=int(loc[0]), x1=int(loc[1]) ,  y2=int(loc[2]), x2=int(loc[3]))
            iaa_bboxes.append(bbox)

        # print(iaa_bboxes)
        # augment image and bounding boxes
        image, bboxes = self.aug(image, iaa_bboxes)
        # bboxes = bboxes[0]


        revised_locs = []
        for loc in bboxes:
            revised_loc = [loc.y1, loc.x1, loc.y2, loc.x2]

            revised_loc[0] = max(0.0,revised_loc[0])/(self.resize[0]-1)
            revised_loc[1] = max(0.0,revised_loc[1])/(self.resize[1]-1)
            revised_loc[2] = min(self.resize[0]-1,revised_loc[2])/(self.resize[0]-1)
            revised_loc[3] = min(self.resize[1]-1,revised_loc[3])/(self.resize[1]-1)
        
            revised_locs.append(revised_loc)
        

        image = self.transimg(image)

        return image, label, revised_locs
        # except:
        #     self.__getitem__(i=random.randint(0,self.__len__()-1))

    def transimg(self, img):
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        return img


    def getiaaAug(self, phase):
        if phase == 'train':
            return iaa.Sequential([
                        iaa.Resize({"height": int(self.resize[0]/0.875), "width": int(self.resize[1]/0.875)}),
                        iaa.CropToFixedSize(width=self.resize[0], height=self.resize[1]),
                        iaa.Fliplr(0.5),
                        ])
        else:
            return iaa.Sequential([
                        iaa.Resize({"height": self.resize[0], "width": self.resize[1]}),
                    ])
    
    def aug(self, image, bboxes=None):
        if bboxes is None:
            weak_imgs = self.iaaSeq(image=image, bounding_boxes=None)
            return weak_imgs
        else:
            weak_imgs, weak_bboxes = self.iaaSeq(image=image, bounding_boxes=bboxes)
            return weak_imgs, weak_bboxes



    def __len__(self):
        return len(self.image_ids)



def collate_fn(batch):
    """
    args:
        batch: [[image, label, img_locs] for seq in batch]
        img_locs = [[4 items] * N]
    return:
        [[image]] * batch_size, [[label]]*batch_szie, [[pad_img_locs]]*batch_szie
    """
 
 
    max_len = 50
 
    lens = [len(dat[-1]) for dat in batch]
 
    max_len = max(lens)
 
    out_image = list()
    out_label = list()
    out_img_locs = list()
    for dat in batch:
        image, label, img_loc = dat
 
        padding = [[0.0, 0.0, 0.0, 0.0] for _ in range(max_len - len(img_loc))]
        img_loc.extend(padding)
 
        out_image.append(image.tolist())
        out_label.append(label)
        out_img_locs.append(img_loc)

 
    out_image = torch.tensor(out_image).float()
    out_label = torch.tensor(out_label).float()
    out_img_locs = torch.tensor(out_img_locs).float()
 

    return out_image, out_label, out_img_locs



if __name__ == '__main__':
    
    from torch.utils.data import DataLoader
    ds = LCHPInitDataset(phase='val',dataset_path='/root/data/',dataset='TINY', meta_path='metadata', y0x0y1x1=True, resize=(64,64))


    dl = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_fn)
    for batch in dl:
        # print(batch)
        image, label,  img_locs = batch

        print(image.shape)
        print(label.shape)
        print(img_locs.shape)
        print(img_locs[-1])
        
        break
