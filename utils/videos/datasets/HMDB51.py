import os
import glob

from itertools import zip_longest

import torch
from torch.utils.data.dataset import Dataset
from torchvision.datasets.folder import default_loader


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def make_videodataset(txt_annotation, frames_path, n_frames, set_number, class_to_idx):
    """
    Creates a dataset from the video frames or optical flow of the HMDB51 dataset.

    Parameters
    ----------
    annotation_path: list of str,
        list of the path to the annotation files.
    frames_path: str,
            path to the folder containing the frames in .jpg format.
    n_frames: int,
        number of frames per video. frames will be extracted uniformly along the video length.
    set_number: str,
        index of the image set. "1"->training, "2"->test, "0"->validation/ignored
    class_to_idx: dict,
        mapping between class names and class numbers.

    Returns
    -------
    images: list (str, int),
        contains the data/target pairs. data is a list of tuples in the form (path to frame, label).

    NOTE: the function is inspired to the make_dataset function in torchvision.datasets.folder.
    See here for more info: https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """

    images = []
    n_videos = 0
    for txt in txt_annotation:
        current_class = os.path.split(txt)[1].split("_test")[0]

        with open(txt, "r") as file:
            for line in file.readlines():

                line = line.rsplit()
                video_name, video_ext = os.path.splitext(line[0])

                set_id = line[1]
                if set_id == set_number:
                    video_path = os.path.join(frames_path, video_name)
                    frames_list = sorted(glob.glob(glob.escape(video_path) + "/*.jpg"))
                    if len(frames_list) < n_frames:
                        frames_list = frames_list * (n_frames // len(frames_list)) + frames_list[
                                                                                     0:n_frames % len(frames_list)]

                    item = (frames_list, class_to_idx[current_class])

                    # item = (accepted_frames, class_to_idx[target])
                    images.append(item)
                    n_videos += 1

    return images, n_videos


class HMDB51Frames3D(Dataset):
    """Dataset object for the HMDB51 dataset. Thought to be used with a 3D CNN."""

    def __init__(self, frames_path, annotation_path, frames_per_clip, step_between_clips, set_type,
                 loader=default_loader, fold=1, transform=None, target_transform=None):
        """
        Parameters
        ----------

        frames_path: str,
            path to the folder containing the frames in .jpg format.
        annotation_path: str,
            path to the folder containing the annotation .txt files.
        frames_per_clip: int,
            number of frames per clip.
        step_between_clips: int,
            distance in frames between two consecutive clips of the same video.
        set_type: str,
            choose between train, test, val.
        loader: Pytorch loader,
            loader for the images. Defaults to the default loader (PIL).
        fold: int,
            fold for the dataset. Choose between 1, 2, 3.
        transform: torch.transform or None,
            transformation to apply to the samples.
        target_transform: torch.transform,
            transformation to apply to the targets (i.e. labels).
        """
        self.dataset_name = "hmdb51"
        self.mode = "rgb"
        self.annotation_path = annotation_path
        self.frames_path = frames_path

        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        self.set_type = set_type
        self.fold = fold

        self.transform = transform
        self.target_transform = target_transform

        self.txt_annotation = self._get_txt_annotations()

        classes, class_to_idx = self._find_classes()
        samples, n_videos = make_videodataset(self.txt_annotation, self.frames_path,
                                              self.frames_per_clip, self.set_to_number(), class_to_idx)
        self.video_targets = {k: samples[k][1] for k in range(len(samples))}
        self.samples, self.clips_idx = self.create_clips(samples)

        self.n_videos = n_videos

        self.loader = loader
        self.classes = classes
        self.class_to_idx = class_to_idx

    def set_to_number(self):
        """

        Returns
        -------
        A mapping between "train", "test" "val" to "1", "2", "0"
        """
        if self.set_type == "train":
            return "1"
        if self.set_type == "test":
            return "2"
        if self.set_type == "val":
            return "0"

    def _get_txt_annotations(self):
        """ returns the appropriate txt files containing the train/test videos for the chosen fold."""
        fold_txt = [os.path.join(self.annotation_path, file) for file in os.listdir(self.annotation_path) if
                    file.split(".txt")[0][-1] == str(self.fold)]
        return fold_txt

    def _find_classes(self):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """

        classes = [os.path.split(txt)[1].split("_test")[0] for txt in self.txt_annotation]

        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx

    def create_clips(self, samples):
        """
        Creates clips from the frames of each videos. Clips will have frames_per_clip frames and be separated by
        frames_per_clip.

        Parameters
        ----------
        samples: list of tuples (list, int),
            contains the frames for each video and the relative label.

        Returns
        -------
        clips: list of tuples (list, int),
            contains the clips of each video
        clips_idx: int,
            index which tells to which video the clip belongs to. Clip i corresponds to video clip_idx[i].
        """

        clips, clips_idx = [], []

        for sample_idx, sample in enumerate(samples):
            clip_start = 0
            frames_path, clip_label = sample[0], sample[1]

            while clip_start < len(frames_path):

                clips_to_add = frames_path[clip_start:clip_start + self.frames_per_clip]
                if len(clips_to_add) == self.frames_per_clip:
                    # The exception is needed to avoid picking a shorter clip at the end of the video.
                    clips.append((clips_to_add, clip_label))
                    clips_idx.append(sample_idx)

                clip_start += self.frames_per_clip + self.step_between_clips

        return clips, clips_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """ The returned samples will be in the shape (3, frames_per_clip, height, width) """

        sample, target = self.samples[index]
        sample = [self.loader(item) for item in sample]
        if self.transform is not None:
            sample = [self.transform(item) for item in sample]

        sample = torch.stack(sample, dim=1)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __info__(self):
        print("{}-{} {} set\t Split = {}".format(self.dataset_name.upper(), self.mode, self.set_type, self.fold))
        print("\tNumber of videos = {}\t ".format(self.n_videos))
        print("\tNumber of clips = {} ({} frames per video separated by {} frames)"
              .format(len(self.samples), self.frames_per_clip, self.step_between_clips))


class HMDB51Flow3D(Dataset):
    """Dataset object for the HMDB51 dataset. Thought to be used with a 3D CNN."""

    def __init__(self, frames_path, annotation_path, frames_per_clip, step_between_clips, set_type,
                 loader=default_loader, fold=1, transform=None, target_transform=None):
        """
        Parameters
        ----------

        frames_path: str,
            path to the folder containing the frames in .jpg format.
        annotation_path: str,
            path to the folder containing the annotation .txt files.
        frames_per_clip: int,
            number of frames per clip.
        step_between_clips: int,
            distance in frames between two consecutive clips of the same video.
        set_type: str,
            choose between train, test, val.
        loader: Pytorch loader,
            loader for the images. Defaults to the default loader (PIL).
        fold: int,
            fold for the dataset. Choose between 1, 2, 3.
        transform: torch.transform or None,
            transformation to apply to the samples.
        target_transform: torch.transform,
            transformation to apply to the targets (i.e. labels).
        """
        self.dataset_name = "hmdb51"
        self.mode = "flow"
        self.annotation_path = annotation_path
        self.frames_path = frames_path

        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        self.set_type = set_type
        self.fold = fold

        self.transform = transform
        self.target_transform = target_transform

        self.txt_annotation = self._get_txt_annotations()

        classes, class_to_idx = self._find_classes()

        samples_u, n_videos_u = make_videodataset(self.txt_annotation, os.path.join(self.frames_path, "u"),
                                                  self.frames_per_clip, self.set_to_number(), class_to_idx)
        samples_v, n_videos_v = make_videodataset(self.txt_annotation, os.path.join(self.frames_path, "v"),
                                                  self.frames_per_clip, self.set_to_number(), class_to_idx)

        self.video_targets = {k: samples_u[k][1] for k in range(len(samples_u))}
        self.samples_u, self.clips_idx = self.create_clips(samples_u)
        self.samples_v, self.clips_idx = self.create_clips(samples_v)

        self.n_videos = n_videos_u

        self.loader = loader
        self.classes = classes
        self.class_to_idx = class_to_idx

    def set_to_number(self):
        """

        Returns
        -------
        A mapping between "train", "test" "val" to "1", "2", "0"
        """
        if self.set_type == "train":
            return "1"
        if self.set_type == "test":
            return "2"
        if self.set_type == "val":
            return "0"

    def _get_txt_annotations(self):
        """ returns the appropriate txt files containing the train/test videos for the chosen fold."""
        fold_txt = [os.path.join(self.annotation_path, file) for file in os.listdir(self.annotation_path) if
                    file.split(".txt")[0][-1] == str(self.fold)]
        return fold_txt

    def _find_classes(self):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """

        classes = [os.path.split(txt)[1].split("_test")[0] for txt in self.txt_annotation]

        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx

    def create_clips(self, samples):
        """
        Creates clips from the frames of each videos. Clips will have frames_per_clip frames and be separated by
        frames_per_clip.

        Parameters
        ----------
        samples: list of tuples (list, int),
            contains the frames for each video and the relative label.

        Returns
        -------
        clips: list of tuples (list, int),
            contains the clips of each video
        clips_idx: int,
            index which tells to which video the clip belongs to. Clip i corresponds to video clip_idx[i].
        """

        clips, clips_idx = [], []

        for sample_idx, sample in enumerate(samples):
            clip_start = 0
            frames_path, clip_label = sample[0], sample[1]

            while clip_start < len(frames_path):

                clips_to_add = frames_path[clip_start:clip_start + self.frames_per_clip]
                if len(clips_to_add) == self.frames_per_clip:
                    # The exception is needed to avoid picking a shorter clip at the end of the video.
                    clips.append((clips_to_add, clip_label))
                    clips_idx.append(sample_idx)

                clip_start += self.frames_per_clip + self.step_between_clips

        return clips, clips_idx

    def __len__(self):
        return len(self.samples_u)

    def __getitem__(self, index):
        """ The returned samples will be in the shape (3, frames_per_clip, height, width) """

        sample_u, target = self.samples_u[index]
        sample_v, _ = self.samples_v[index]

        sample_u = [self.loader(item) for item in sample_u]
        sample_v = [self.loader(item) for item in sample_v]

        if self.transform is not None:
            sample_u = [self.transform(item) for item in sample_u]
            sample_v = [self.transform(item) for item in sample_v]

        sample_u = torch.stack(sample_u, dim=1)
        sample_v = torch.stack(sample_v, dim=1)

        sample = torch.cat((sample_u, sample_v), 0)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __info__(self):
        print("{}-{} {} set\t Split = {}".format(self.dataset_name.upper(), self.mode, self.set_type, self.fold))
        print("\tNumber of videos = {}\t ".format(self.n_videos))
        print("\tNumber of clips = {} ({} frames per video separated by {} frames)"
              .format(len(self.samples_u), self.frames_per_clip, self.step_between_clips))
