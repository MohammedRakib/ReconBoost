import torch
import os
import numpy as np
from torch.utils.data import Dataset
import csv
import random
import librosa
import torch.nn.functional
from PIL import Image
from torchvision import transforms

class VGGSound(Dataset):

    def __init__(self, mode='train'):
        self.mode = mode
        train_video_data = []
        train_audio_data = []
        test_video_data = []
        test_audio_data = []
        train_label = []
        test_label = []
        train_class = []
        test_class = []

        with open('/home/rakib/Multi-modal-Imbalance/data/VGGSound/vggsound.csv') as f:
            csv_reader = csv.reader(f)
            next(csv_reader)  # Skip the header
            
            for item in csv_reader:
                youtube_id = item[0]
                timestamp = "{:06d}".format(int(item[1]))  # Zero-padding the timestamp
                train_test_split = item[3]

                video_dir = os.path.join('/home/rakib/Multimodal-Datasets/VGGSound/video/frames', train_test_split, 'Image-{:02d}-FPS'.format(1), f'{youtube_id}_{timestamp}')
                audio_dir = os.path.join('/home/rakib/Multimodal-Datasets/VGGSound/audio', train_test_split, f'{youtube_id}_{timestamp}.wav')

                if os.path.exists(video_dir) and os.path.exists(audio_dir) and len(os.listdir(video_dir)) > 3:
                    if train_test_split == 'train':
                        train_video_data.append(video_dir)
                        train_audio_data.append(audio_dir)
                        if item[2] not in train_class: 
                            train_class.append(item[2])
                        train_label.append(item[2])
                    elif train_test_split == 'test':
                        test_video_data.append(video_dir)
                        test_audio_data.append(audio_dir)
                        if item[2] not in test_class: 
                            test_class.append(item[2])
                        test_label.append(item[2])

        self.classes = train_class
        class_dict = dict(zip(self.classes, range(len(self.classes))))

        if mode == 'train':
            self.video = train_video_data
            self.audio = train_audio_data
            self.label = [class_dict[label] for label in train_label]
        elif mode == 'test':
            self.video = test_video_data
            self.audio = test_audio_data
            self.label = [class_dict[label] for label in test_label]

    def __len__(self):
        return len(self.video)

    def __getitem__(self, idx):
        # Audio processing (using librosa to compute the spectrogram)
        sample, rate = librosa.load(self.audio[idx], sr=16000, mono=True)
        while len(sample) / rate < 10.:
            sample = np.tile(sample, 2)

        start_point = random.randint(0, rate * 5)
        new_sample = sample[start_point:start_point + rate * 5]
        new_sample[new_sample > 1.] = 1.
        new_sample[new_sample < -1.] = -1.

        spectrogram = librosa.stft(new_sample, n_fft=256, hop_length=128)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)

        # Image transformations based on mode
        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Image processing
        image_samples = os.listdir(self.video[idx])
        image_samples = sorted(image_samples)
        pick_num = 3  # Fixed number of frames to match AVDataset's behavior
        seg = int(len(image_samples) / pick_num)
        image_arr = []

        for i in range(pick_num):
            tmp_index = int(seg * i)
            img = Image.open(os.path.join(self.video[idx], image_samples[tmp_index])).convert('RGB')
            img = transform(img)
            img = img.unsqueeze(1).float()  # Add channel dimension for concatenation
            image_arr.append(img)
            if i == 0:
                image_n = img
            else:
                image_n = torch.cat((image_n, img), 1)  # Concatenate along the channel dimension

        # Label
        label = self.label[idx]

        return spectrogram, image_n, label