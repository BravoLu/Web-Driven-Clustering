import os
from PIL import Image, ImageFilter
import pdb

class Preprocessor(object):
    def __init__(self, dataset,name,training=True, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.training = training
        self.transform = transform
        self.name = name
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.name in ['WebSearchVehicle']:
            imgPath, typeID = self.dataset[index]
            img =Image.open(imgPath)

            img = img.convert('RGB')

            if self.transform is not None:
                img = self.transform(img)

            return img, typeID

        if self.name in ['Pseudo']:
            imgPath, typeID, fname = self.dataset[index]
            img =Image.open(imgPath)

            img = img.convert('RGB')

            if self.transform is not None:
                img = self.transform(img)

            return img, typeID, fname

        if self.name in ['VehicleID']:
            if self.training:
                imgPath, ID, camID, modelID, colorID= self.dataset[index]
                fname = os.path.basename(imgPath).split('.')[0]
                img = Image.open(imgPath).convert('RGB')

                if self.transform is not None:
                    img = self.transform(img)

                return img, ID, camID, modelID, colorID, fname
            else:
                imgPath, ID, camID = self.dataset[index]
                fname = os.path.basename(imgPath).split('.')[0]
                img = Image.open(imgPath).convert('RGB')

                if self.transform is not None:
                    img = self.transform(img)

                return img, ID, camID, fname

        if self.name in ['VeRi']:
            if self.training:
                imgPath, ID, camID, typeID, colorID, fname = self.dataset[index]
                img = Image.open(imgPath).convert('RGB')

                if self.transform is not None:
                    img = self.transform(img)

                return img,ID,camID,typeID,colorID,fname
            else:
                imgPath, ID, camID, fname = self.dataset[index]
                img = Image.open(imgPath).convert('RGB')

                if self.transform is not None:
                    img = self.transform(img)

                return img, ID, camID, fname

        if self.name in ['VeRi_Wild']:
            if self.training:
                imgPath, ID, camID, modelID, colorID, typeID, fname= self.dataset[index]
                #fname = os.path.basename(imgPath).split('.')[0]
                img = Image.open(imgPath).convert('RGB')

                if self.transform is not None:
                    img = self.transform(img)

                return img, ID, camID, modelID, colorID, typeID, fname
            else:
                imgPath, ID, camID, modelID, colorID, typeID, fname = self.dataset[index]
                #fname = os.path.basename(imgPath).split('.')[0]
                img = Image.open(imgPath).convert('RGB')

                if self.transform is not None:
                    img = self.transform(img)

                return img, ID, camID, fname

        if self.name in ['VeRi_Wild_ALL']:
            imgPath = self.dataset[index][0]
            img = Image.open(imgPath).convert('RGB')
            img = self.transform(img)

            return [img]





