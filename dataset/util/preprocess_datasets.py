import os
import pickle
from PIL import Image
import h5py
import json

# dowload raw data
# from torchmeta.datasets import Omniglot, DoubleMNIST, TripleMNIST, CUB, FC100
# dataset = Omniglot("./", meta_split='train', download=True)


if __name__ == "__main__":
    import numpy as np
    import pickle
    import io

    def process_cifar100_dataset(split, name=""):
        """
        Format h5py.
        Each key is a class. Values are all the images with that class.
        """
        data_file = h5py.File("./" + name + "/" + "data.hdf5", 'r')
        
        data_resized = {}
        with open("./" + name + "/fc100/" + split + "_labels.json", 'r') as f:
            classes = json.load(f)
        
        for cl in classes:
            print(cl)
            data = data_file[cl[0]]
            images = data[cl[1]]
            
            tmp = []
            for i in range(images.shape[0]):
                img = Image.fromarray(images[i])
                tmp.append(np.array(img))

            tmp = np.stack(tmp, 0)
            data_resized[cl[1]] = tmp

        with open(split + "_" + name + ".pkl", 'wb') as f:
            pickle.dump(data_resized, f)

    def process_rgb_dataset(split, name="", size=64):
        """
        Minimagenet and CUB.
        Format h5py.
        Each key is a class. Values are all the samples for that class.
        """
        data_file = h5py.File("./" + name + "/" + split + "_data.hdf5", 'r')
        data = data_file['datasets']
        
        data_resized = {}
        classes = data.keys()

        for cl in classes:
            print(cl)
            images = data[cl]
            
            tmp = []
            for i in range(images.shape[0]):
                if name == "cub":
                    img = Image.open(io.BytesIO(images[i])).convert('RGB')
                else:
                    img = Image.fromarray(images[i])
                
                img_resized = img.resize((size, size), Image.BOX)
                tmp.append(np.array(img_resized))
                
            tmp = np.stack(tmp, 0)
            data_resized[cl] = tmp

        with open(split + "_" + name + ".pkl", 'wb') as f:
            pickle.dump(data_resized, f)

    
    def process_omniglot_dataset(split, name=""):
        """
        Format h5py.
        Omniglot different format than the other binary datasets.
        """
        data_file = h5py.File("./" + name + "/" + "data.hdf5", 'r')
        if split == "train":
            data = data_file["images_background"]
        elif split == "test":
            data = data_file["images_evaluation"]
        else:
            print("No validation for Omniglot")

        data_resized = {}
        print(data.keys())
        alphabets = data.keys()

        c = 0
        for alphabet in alphabets:
            data_alphabet = data[alphabet]
        
            classes = data_alphabet.keys()
            for cl in classes:
                print(cl)
                images = data_alphabet[cl]
                
                tmp = []
                for i in range(images.shape[0]):
                    img = Image.fromarray(images[i])
                    img_resized = img.resize((28, 28), Image.BOX)
                    tmp.append(np.array(img_resized))
                    
                tmp = np.stack(tmp, 0)
                data_resized[c] = tmp
                c += 1

        with open(split + "_" + name + ".pkl", 'wb') as f:
            pickle.dump(data_resized, f)

    def process_binary_dataset(split, name=""):
        """
        Format h5py.
        Process doubleMNIST and tripleMNIST.
        Each key is a class. Values are all the images with that class.
        For binary datasets, need to convert it back. 
        """
        data_file = h5py.File("./" + name + "/" + split + "_data.hdf5", 'r')
        data = data_file['datasets']
        
        data_resized = {}
        print(data.keys())
        
        classes = data.keys()
        for cl in classes:
            print(cl)
            images = data[cl]
            
            tmp = []
            for i in range(images.shape[0]):
                img = Image.open(io.BytesIO(images[i])).convert('L')#.convert('L')
                img_resized = img.resize((28, 28), Image.BOX)
                tmp.append(np.array(img_resized))
            tmp = np.stack(tmp, 0)
            data_resized[cl] = tmp

        with open(split + "_" + name + ".pkl", 'wb') as f:
            pickle.dump(data_resized, f)

    #for name in ["doubelmnist", "triplemnist"]:
    name = "cifar100"
    #size = 64
    if name == "omniglot":
        process_omniglot_dataset('train', name)
        process_omniglot_dataset('test', name)
    elif name in ["doubelmnist", "triplemnist"]:
        process_binary_dataset('train', name)
        process_binary_dataset('val', name)
        process_binary_dataset('test', name)
    elif name in ["minimagenet", "cub"]:
        process_rgb_dataset('train', name, size)
        process_rgb_dataset('val', name, size)
        process_rgb_dataset('test', name, size)
    elif name in ["cifar100"]:
        process_cifar100_dataset('train', name)
        process_cifar100_dataset('val', name)
        process_cifar100_dataset('test', name)

    # print("test")
    # process_minimagenet('test')
    # with open("train_minimagenet.pkl", 'rb') as f:
    #     data = pickle.load(f)
    
    # tmp = data['n03476684'][0]
    # print(tmp / 255.)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def process_cifar100_dataset(split, name=""):
    """
    Format h5py.
    Each key is a class. Values are all the images with that class.
    """
    data_file = h5py.File("./" + name + "/" + "data.hdf5", 'r')

    data_dict=unpickle(split)
    labels = data_dict[b'fine_labels']
    data = data_dict[b'data']

    data_out = {}

    for i in range(data.shape[0]):
        cl = labels[i]
        if cl not in data_out:
            data_out[cl] = []
        image = np.array(Image.fromarray(data[i]))
    
    data_resized = {}
    with open("./" + name + "/fc100/" + split + "_labels.json", 'r') as f:
        classes = json.load(f)
    
    for cl in classes:
        print(cl)
        data = data_file[cl[0]]
        images = data[cl[1]]
        
        tmp = []
        for i in range(images.shape[0]):
            img = Image.fromarray(images[i])
            tmp.append(np.array(img))

        tmp = np.stack(tmp, 0)
        data_resized[cl[1]] = tmp

    with open(split + "_" + name + ".pkl", 'wb') as f:
        pickle.dump(data_resized, f)


import numpy as np
import h5py
import json
from PIL import Image
import pickle
import os
import tqdm
def process_cifar100_dataset(split, name=""):
    """
    Format h5py.
    Each key is a class. Values are all the images with that class.
    """
    data_file = h5py.File("./" + name + "/" + "data.hdf5", 'r')
    data_resized = {}
    with open("./" + name + "/fc100/" + split + "_labels.json", 'r') as f:
        classes = json.load(f)
    for cl in classes:
        print(cl)
        data = data_file[cl[0]]
        images = data[cl[1]]
        tmp = []
        for i in range(images.shape[0]):
            img = Image.fromarray(images[i])
            tmp.append(np.array(img))
        tmp = np.stack(tmp, 0)
        data_resized[cl[1]] = tmp
    with open(split + "_" + name + ".pkl", 'wb') as f:
        pickle.dump(data_resized, f)


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

def from_folder_by_category(source_root):
    source_paths = sorted(make_dataset(source_root))
    all_category_paths = {}
    for path in source_paths:
        cate = path.split('/')[-1].split('_')[0]
        if cate not in all_category_paths:
            all_category_paths[cate] = []
        all_category_paths[cate].append(path)
    return all_category_paths

def process_dataset_from_folder(folder, size, split, name):
    source_paths = sorted(make_dataset(folder))
    images_by_class = {}
    for path in tqdm.tqdm(source_paths):
        cate = path.split('/')[-1].split('_')[0]
        if cate not in images_by_class:
            images_by_class[cate] = []
        img = Image.open(path).convert('RGB').resize((size, size))
        images_by_class[cate].append(np.array(img))
    for cls in images_by_class.keys():
        images_by_class[cls] = np.stack(images_by_class[cls], 0)
    with open(split + "_" + name + ".pkl", 'wb') as f:
        pickle.dump(images_by_class, f)

