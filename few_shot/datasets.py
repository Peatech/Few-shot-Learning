from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms
from skimage import io
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

from config import DATA_PATH
"""
The OmniglotDataset class represents the Omniglot dataset, a common dataset used for few-shot learning. 
The class provides a structured way to:
Index the dataset: Collect metadata about images and organize them in a DataFrame.
Convert images into PyTorch-compatible format: Load and normalize images, and return them as tensors along with their labels.
Support dataset querying: Enable functionality like len(dataset) and dataset[index].

The OmniglotDataset class provides an interface to load and preprocess Omniglot images for PyTorch models.
Data Structure:
Metadata is stored in a Pandas DataFrame (self.df) for efficient querying and management.
Workflow:
Images are indexed (file paths, class names, IDs).
Images are retrieved, normalized, and returned as tensors along with labels.
Integration: This class enables tasks like:
Counting total images: len(dataset)
Fetching an image and label: dataset[index]

"""

class OmniglotDataset(Dataset):
    def __init__(self, subset):
        """
        The constructor validates the subset argument, which should be either 'background' (training set) or 
        'evaluation' (test set).
        """
        if subset not in ('background', 'evaluation'):
            raise(ValueError, 'subset must be one of (background, evaluation)')
        self.subset = subset

        # The static method index_subset collects information (file paths, class names, etc.) about all images in the specified subset:
        """
        Calls the static method index_subset to traverse the filesystem and collect metadata about each image in the specified subset.
        Stores this metadata in a Pandas DataFrame (self.df) for efficient querying and management.

        """
        self.df = pd.DataFrame(self.index_subset(self.subset))

        # Each image in the dataset is assigned a unique ID, stored in the id column
        self.df = self.df.assign(id=self.df.index.values)

        # Each class name is mapped to an integer ID (0 to num_classes−1). 
        self.unique_characters = sorted(self.df['class_name'].unique())
        
        # A new column class_id is created, storing the integer ID for each image’s class
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        # Metadata is converted into dictionaries for faster access:
        self.datasetid_to_filepath = self.df.to_dict()['filepath'] # Maps image IDs to file paths.
        self.datasetid_to_class_id = self.df.to_dict()['class_id'] # Maps image IDs to class labels.

    def __getitem__(self, item):    # This method retrieves an image and its label given an index
        instance = io.imread(self.datasetid_to_filepath[item])
        # Reindex to channels first format as supported by pytorch
        instance = instance[np.newaxis, :, :]

        # Normalise to 0-1
        instance = (instance - instance.min()) / (instance.max() - instance.min())

        label = self.datasetid_to_class_id[item]    # The corresponding label for the image is retrieved

        return torch.from_numpy(instance), label    # The image and label are returned as a PyTorch tensor and an integer, respectively

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    @staticmethod
    def index_subset(subset):
        """
        This static method generates metadata for all images in a specified subset (background or evaluation).

        For each image:
        
        Class Information:
        The folder structure encodes the alphabet (parent folder) and class_name (subfolder):
        alphabet = root.split('/')[-2]
        class_name = '{}.{}'.format(alphabet, root.split('/')[-1])
        File Path:
        The full path to the image file is stored:
        'filepath': os.path.join(root, f)
        Other Details:
        Additional metadata like subset and alphabet are also stored.
        3. Progress Bar
        
        The tqdm progress bar tracks the indexing process:
        
        progress_bar = tqdm(total=subset_len)
        4. Return Metadata
        
        The method returns a list of dictionaries, where each dictionary contains metadata for one image:
        
        return images

        
        """
        images = []
        print('Indexing {}...'.format(subset))
        # Quick first pass to find total for tqdm bar
        subset_len = 0
        for root, folders, files in os.walk(DATA_PATH + '/Omniglot/images_{}/'.format(subset)):
            subset_len += len([f for f in files if f.endswith('.png')])

        progress_bar = tqdm(total=subset_len)
        for root, folders, files in os.walk(DATA_PATH + '/Omniglot/images_{}/'.format(subset)):
            if len(files) == 0:
                continue

            alphabet = root.split('/')[-2]
            class_name = '{}.{}'.format(alphabet, root.split('/')[-1])

            for f in files:
                progress_bar.update(1)
                images.append({
                    'subset': subset,
                    'alphabet': alphabet,
                    'class_name': class_name,
                    'filepath': os.path.join(root, f)
                })

        progress_bar.close()
        return images


class MiniImageNet(Dataset):
    def __init__(self, subset):
        """Dataset class representing miniImageNet dataset

        # Arguments:
            subset: Whether the dataset represents the background or evaluation set
        """
        if subset not in ('background', 'evaluation'):
            raise(ValueError, 'subset must be one of (background, evaluation)')
        self.subset = subset

        self.df = pd.DataFrame(self.index_subset(self.subset))

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']

        # Setup transforms
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.Resize(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, item):
        instance = Image.open(self.datasetid_to_filepath[item])
        instance = self.transform(instance)
        label = self.datasetid_to_class_id[item]
        return instance, label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    @staticmethod
    def index_subset(subset):
        """Index a subset by looping through all of its files and recording relevant information.

        # Arguments
            subset: Name of the subset

        # Returns
            A list of dicts containing information about all the image files in a particular subset of the
            miniImageNet dataset
        """
        images = []
        print('Indexing {}...'.format(subset))
        # Quick first pass to find total for tqdm bar
        subset_len = 0
        for root, folders, files in os.walk(DATA_PATH + '/miniImageNet/images_{}/'.format(subset)):
            subset_len += len([f for f in files if f.endswith('.png')])

        progress_bar = tqdm(total=subset_len)
        for root, folders, files in os.walk(DATA_PATH + '/miniImageNet/images_{}/'.format(subset)):
            if len(files) == 0:
                continue

            class_name = root.split('/')[-1]

            for f in files:
                progress_bar.update(1)
                images.append({
                    'subset': subset,
                    'class_name': class_name,
                    'filepath': os.path.join(root, f)
                })

        progress_bar.close()
        return images


class DummyDataset(Dataset):
    def __init__(self, samples_per_class=10, n_classes=10, n_features=1):
        """Dummy dataset for debugging/testing purposes

        A sample from the DummyDataset has (n_features + 1) features. The first feature is the index of the sample
        in the data and the remaining features are the class index.

        # Arguments
            samples_per_class: Number of samples per class in the dataset
            n_classes: Number of distinct classes in the dataset
            n_features: Number of extra features each sample should have.
        """
        self.samples_per_class = samples_per_class
        self.n_classes = n_classes
        self.n_features = n_features

        # Create a dataframe to be consistent with other Datasets
        self.df = pd.DataFrame({
            'class_id': [i % self.n_classes for i in range(len(self))]
        })
        self.df = self.df.assign(id=self.df.index.values)

    def __len__(self):
        return self.samples_per_class * self.n_classes

    def __getitem__(self, item):
        class_id = item % self.n_classes
        return np.array([item] + [class_id]*self.n_features, dtype=np.float), float(class_id)
