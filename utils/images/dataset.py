import os
import glob

from torch.utils.data.dataset import Dataset

from torchvision.datasets.folder import default_loader


def make_animalsdataset(path, test_ratio, class_to_idx):
    train_samples, test_samples = [], []

    for class_folder in os.listdir(path):
        class_images = sorted(glob.glob(os.path.join(path, class_folder, "*.jpeg")))
        n_train_images = int(len(class_images) * (1 - test_ratio / 100))

        train_samples.extend([(image, class_to_idx[class_folder]) for image in class_images[:n_train_images]])
        test_samples.extend([(image, class_to_idx[class_folder]) for image in class_images[n_train_images:]])

    return train_samples, test_samples


class Animals10(Dataset):

    def __init__(self, dataset_path, test_ratio, mode="train", loader=default_loader,
                 transform=None, target_transform=None):
        """
        Parameters
        ----------
        dataset_path: str,
            path to the folder containing the frames in .jpg format.
        annotation_path: str,
            path to the folder containing the annotation .txt files.
        set_type: str,
            choose between trainval, test.
        loader: Pytorch loader,
            loader for the images. Defaults to the default loader (PIL).
        transform: torch.transform,
            transformation to apply to the samples.
        target_transform: torch.transform,
            transformation to apply to the targets (i.e. labels).
        """

        self.dataset_path = dataset_path
        self.test_ratio = test_ratio
        self.mode = mode

        self.transform = transform
        self.target_transform = target_transform
        self.class_idx = self.class_to_idx()

        samples_train, samples_test = make_animalsdataset(self.dataset_path, self.test_ratio, self.class_idx)

        samples = samples_train if self.mode == "train" else samples_test

        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.loader = loader
        self.classes = [i for i in range(10)]

    def __len__(self):
        return len(self.samples)

    def class_to_idx(self):
        class_idx = {class_folder: i for i, class_folder in enumerate(os.listdir(self.dataset_path))}
        return class_idx

    def __getitem__(self, index):

        sample, target = self.samples[index]
        sample = self.loader(sample)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
