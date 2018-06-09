import os

import torch
import numpy as np
import cv2 as cv
import fnmatch


class Prepare():
    def __init__(self, args):
        self.args = args
        pass

    # get person directory
    def getPersonDirsList(self, rgb_dirDir):
        if self.args.dataset == 1 or self.args.dataset == 0:
            firstCameraDirName = 'cam1'
        else:
            firstCameraDirName = 'cam_a'
        seqDir = rgb_dirDir + '/' + firstCameraDirName
        personDir = os.listdir(seqDir)
        if len(personDir) == 0:
            return False
        for i, dir in enumerate(personDir):
            if len(dir) <= 2:
                del personDir[i]
        personDir.sort()
        return personDir

    # get name list of images
    def getSequenceImages(self, rgb_dir, file_suffix):
        img_list = os.listdir(rgb_dir)
        img = fnmatch.filter(img_list, '*.' + file_suffix)
        img.sort()
        return img

    # load images into tensor
    def loadSequenceImages(self, rgb_dir, of_dir, img_list):
        nImgs = len(img_list)
        print(nImgs, rgb_dir, of_dir)
        imagePixelData = torch.zeros((nImgs, 5, 64, 48), dtype=torch.float32,
                                     device=torch.device('cuda', 0))
        for i, file in enumerate(img_list):
            filename = '/'.join([rgb_dir, file])
            filename_of = '/'.join([of_dir, file])
            img = cv.imread(filename)
            img = cv.resize(img, (48, 64))
            img = img.astype(np.float32)

            imgof = cv.imread(filename_of)
            imgof = cv.resize(imgof, (48, 64))
            imgof = imgof.astype(np.float32)

            # change image to YUV channels
            img = cv.cvtColor(img, cv.COLOR_BGR2YUV)
            img_tensor = torch.from_numpy(img).type(torch.float32)
            imgof_tensor = torch.from_numpy(imgof).type(torch.float32)
            for c in range(3):
                v = torch.sqrt(torch.var(img_tensor[:, :, c]))
                m = torch.mean(img_tensor[:, :, c])
                img_tensor[:, :, c] = img_tensor[:, :, c] - m
                img_tensor[:, :, c] = img_tensor[:, :, c] / torch.sqrt(v)
                imagePixelData[i, c] = img_tensor[:, :, c]

            for c in range(2):
                v = torch.sqrt(torch.var(imgof_tensor[:, :, c]))
                m = torch.mean(imgof_tensor[:, :, c])
                imgof_tensor[:, :, c] = imgof_tensor[:, :, c] - m
                imgof_tensor[:, :, c] = imgof_tensor[:, :, c] / torch.sqrt(v)
                imagePixelData[i, c + 3] = imgof_tensor[:, :, c]
                if self.args.disableOpticalFlow == 1:
                    imagePixelData[i, c + 3] = torch.mul(imagePixelData[i, c + 3], 0)

        return imagePixelData

    # index contents of data: data[person_id][camera_id][img_id][channel_id][img_tensor_data]
    def prepareDataset(self, seq_root_rgb, seq_root_of, file_suffix):
        personDir = self.getPersonDirsList(seq_root_rgb)
        dataset = {}
        for i, pdir in enumerate(personDir):
            dataset[i] = {}
            for cam in [1, 2]:
                if self.args.dataset == 1 or self.args.dataset == 0:
                    camera_name = ''.join(['cam', str(cam)])
                elif self.args.dataset == 2:
                    camera_name = ''.join(['cam_', str(cam)])
                rgb_dir = '/'.join([seq_root_rgb, camera_name, pdir])
                of_dir = '/'.join([seq_root_of, camera_name, pdir])
                img_list = self.getSequenceImages(rgb_dir, file_suffix)
                dataset[i][cam] = self.loadSequenceImages(rgb_dir, of_dir, img_list)
        return dataset
