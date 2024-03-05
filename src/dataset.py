import os
import glob
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset
from scipy import signal
from scipy.io import wavfile
import cv2
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader

from src.paths import IAM_RULE_DIR


class Synth90kDataset(Dataset):
    CHARS = '0123456789abcdefghijklmnopqrstuvwxyz'
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__(self, root_dir=None, mode=None, paths=None, img_height=32, img_width=100):
        if root_dir and mode and not paths:
            paths, texts = self._load_from_raw_files(root_dir, mode)
        elif not root_dir and not mode and paths:
            texts = None

        self.paths = paths
        self.texts = texts
        self.img_height = img_height
        self.img_width = img_width

    def _load_from_raw_files(self, root_dir, mode):
        mapping = {}
        with open(os.path.join(root_dir, 'lexicon.txt'), 'r') as fr:
            for i, line in enumerate(fr.readlines()):
                mapping[i] = line.strip()

        paths_file = None
        if mode == 'train':
            paths_file = 'annotation_train.txt'
        elif mode == 'dev':
            paths_file = 'annotation_val.txt'
        elif mode == 'test':
            paths_file = 'annotation_test.txt'

        paths = []
        texts = []
        with open(os.path.join(root_dir, paths_file), 'r') as fr:
            for line in fr.readlines():
                path, index_str = line.strip().split(' ')
                path = os.path.join(root_dir, path)
                index = int(index_str)
                text = mapping[index]
                paths.append(path)
                texts.append(text)
        return paths, texts

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        try:
            image = Image.open(path).convert('L')  # grey-scale
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]

        image = image.resize((self.img_width, self.img_height), resample=Image.BILINEAR)
        image = np.array(image)
        image = image.reshape((1, self.img_height, self.img_width))
        image = (image / 127.5) - 1.0

        image = torch.FloatTensor(image)
        if self.texts:
            text = self.texts[index]
            target = [self.CHAR2LABEL[c] for c in text]
            target_length = [len(target)]

            target = torch.LongTensor(target)
            target_length = torch.LongTensor(target_length)
            return image, target, target_length
        else:
            return image


class Synth90kSample(Dataset):
    """
    This is dataset for a smaller sample of the main MJSynth data
    """
    CHARS = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__(self, root_dir=None, mode=None,split=None,
                 img_height=32, img_width=100, word_len=2):
        self.paths, self.texts = self._load_from_raw_files(root_dir, mode, split,word_len)

        self.img_height = img_height
        self.img_width = img_width

    def __len__(self):
        return len(self.paths)

    def _load_from_raw_files(self, root_dir, mode, split, word_len):
        paths = os.listdir(root_dir)
        length = len(paths)
        if mode=='validation':
            paths = paths[int(split*length):]

        image_paths = list()
        image_texts = list()
        for path in paths:
            image_text = path.split("_")[1]
            if len(image_text) > word_len:
                image_paths.append(root_dir + "/" + path)
                image_texts.append(path.split("_")[1])

        return image_paths, image_texts

    def __getitem__(self, index):
        path = self.paths[index]
        try:
            image = Image.open(path).convert('L')  # grey-scale
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]

        image = image.resize((self.img_width, self.img_height), resample=Image.BILINEAR)
        image = np.array(image)

        image = image.reshape((1, self.img_height, self.img_width))

        image = (image / 127.5) - 1.0

        image = torch.FloatTensor(image)
        if self.texts:
            text = self.texts[index]
            target = [self.CHAR2LABEL[c] for c in text]
            target_length = [len(target)]

            target = torch.LongTensor(target)
            target_length = torch.LongTensor(target_length)
            return image, target, target_length
        else:
            return image


class IAMDataset(Dataset):
    def __init__(self, root_dir=None, mode=None,
                 img_height=32, img_width=100, word_len=2):
        self.paths, self.texts = self._load_from_raw_files(root_dir, mode, word_len)

        self.img_height = img_height
        self.img_width = img_width


class IAMDataset(BaseDataset):
    """
    train_root_dir = '/home/mujahid/PycharmProjects/ssl_wordspotting/data/IEHR/words_training'
    test_root_dir = '/home/mujahid/PycharmProjects/ssl_wordspotting/data/IEHR/words_test'
    train_dataset = IAMDataset(train_root_dir)
    test_dataset = IAMDataset(test_root_dir)
    """
    pass


## images : words/l1/l1-l2/l1-l2-ldx-idx.png : sample
## l1 -  sample_group_root_dir
## l1-l2 - sample_group_dir
## rules : l1-l2-ldx - sample_sub_group
## xml : l1-l2-ldx-idx - sample_name


class IAMDataset2(Dataset):
    CHARS = '0123456789abcdefghijklmnopqrstuvwxyz'
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__(self, root_dir, mode=None, paths=None, img_height=32, img_width=100):
        if root_dir and mode and not paths:
            paths, texts = self._load_from_raw_files(root_dir, mode)
        elif not root_dir and not mode and paths:
            texts = None
        self.paths = paths
        self.texts = texts
        self.img_height = img_height
        self.img_width = img_width

    def _load_from_raw_files(self, root_dir, mode):
        if mode == 'train':
            self.rule_file_path = IAM_RULE_DIR / "trainset.txt"
        elif mode == 'test':
            self.rule_file_path = IAM_RULE_DIR / "testset.txt"
        elif mode == 'val':
            self.rule_file_path = IAM_RULE_DIR / "validationset1.txt"
        self.line_folders = None
        self.line_folders, self.line_dirs = self.create_line_dirs()
        self.samples, self.word_strings = self.get_word_labels()
        self.labels_encoder()
        pass

    def get_unique_word_strings(self):
        unique_word_strings, counts = np.unique(self.word_strings, return_counts=True)
        return unique_word_strings, counts

    def get_query_list(self):
        if self.ttype == 'test':
            unique_word_strings, counts = self.get_unique_word_strings()
            qry_word_ids = unique_word_strings[np.where(counts > 1)[0]]
            query_list = np.zeros(len(self.word_strings), np.int8)
            qry_ids = [i for i in range(len(self.word_strings)) if self.word_strings[i] in qry_word_ids]
            query_list[qry_ids] = 1
            self.query_list = query_list

    def labels_encoder(self):
        labels, _ = self.get_unique_word_strings()
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, img_path, label = self.samples[idx]
        encoded_label = self.label_encoder.transform([label])
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        # is_query = self.query_list[idx]
        # return img_id, img, label, encoded_label[0], is_query
        return img_id, img, label, encoded_label[0]

    def get_xml_file_object(self, path):
        tree = ET.parse(path)
        root = tree.getroot()
        return root

    def get_word_labels(self):
        word_ids, word_paths = self.get_words()

        word_ids, xml_paths = self.construct_xml_file_paths(word_ids)
        ll = list()
        labels = list()
        for word_id, word_path, xml_path in zip(word_ids, word_paths, xml_paths):
            root = self.get_xml_file_object(xml_path)
            for word in root.iter('word'):
                img_id = word.get('id')
                if img_id == word_id:
                    label = word.get('text')
                    ll.append((word_id, word_path, label))
                    labels.append(label)

        return ll, labels

    def construct_xml_file_paths(self, word_ids):
        xml_paths = ['-'.join(i.split('-')[:-2]) + '.xml' for i in word_ids]
        xml_paths = [IAM_XML_DIR / i for i in xml_paths]
        return word_ids, xml_paths

    def get_words(self):
        # print(len(sample_group_dir))

        image_paths = [glob.glob(f"{i}/*.png") for i in self.line_dirs]
        # print(sample_file_paths)

        image_paths = [item for sublist in image_paths for item in sublist]

        line_ids = self.read_line_ids()

        word_paths = [i for i in image_paths if '-'.join(pathlib.Path(i).name.split('-')[:-1])
                      in line_ids]
        word_ids = [pathlib.Path(i).name.split('.')[0] for i in word_paths]

        return word_ids, word_paths
        #

    def create_line_dirs(self):
        # images : words/l1/l1-l2/l1-l2-ldx-idx.png : sample
        # l1 -  sample_group_root_dir
        # l1-l2 - line_folders
        # rules : l1-l2-ldx - line_ids
        # xml file name: l1-l2.xml
        # xml : l1-l2-ldx-idx - sample_name
        line_ids = self.read_line_ids()
        line_folders = [f"{i.split('-')[0]}-{i.split('-')[1]}" for i in line_ids]
        line_folders = list(dict.fromkeys(line_folders))
        line_dirs = [IAM_WORDS_DIR / i.split('-')[0] / f"{i.split('-')[0]}-{i.split('-')[1]}"
                     for i in line_ids]
        line_dirs = list(dict.fromkeys(line_dirs))
        return line_folders, line_dirs

    def read_line_ids(self):
        # this method reads the rules.txt files and
        # returns its contents sample_sub_groups
        with open(self.rule_file_path) as f:
            line_ids = [i.replace('\n', '').strip() for i in f.readlines()]
        logging.info(f"{len(line_ids)} dirs for {self.ttype} set")
        return line_ids

    def image_names(self, dir_path):
        image_names = os.listdir(dir_path)
        return image_names

    def get_random_samples(self, number=9):
        random_samples = random.sample(self.samples, number)
        return random_samples


def synth90k_collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths

# if __name__ == '__main__':
#     dataset = Synth90kSample(root_dir="/home/mujahid/PycharmProjects/crnn-pytorch/data/mjsynth_sample")
#     train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
