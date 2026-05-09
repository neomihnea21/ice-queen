
import numpy as np
import soundfile as sf
import torch.nn as nn
import torch, copy
from torchsummary import summary

# in testing, I coudld get decent-ish results with 30 dictionary atoms
# so here, I'll try and narrow it down to 64

# so here's a downsampling block
class DownBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv = nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size = 3, padding = 1),
        nn.ReLU(),
        nn.Conv1d(out_channels, out_channels, kernel_size = 3, padding = 1),
        nn.ReLU()
    )
    # this is where we halve input length
    self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
  def forward(self, x):
    x = self.conv(x)
    skip = x.clone()
    x = self.pool(x)
    return x, skip



# PRIMO VICTORIA: we have an autoencoder that works, now let's come up with some data for it
class UNet_Autoencoder(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.down1 = DownBlock(in_channels, 32)
    self.down2 = DownBlock(32, 64)
    self.down3 = DownBlock(64, 128)

    self.bottleneck = nn.Sequential(
        nn.Conv1d(128, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv1d(256, 256, kernel_size=3, padding=1),
        nn.ReLU()
    )

    self.final_layer = nn.Conv1d(32, 1, kernel_size=1)
  def forward(self, x):
    x, skip1 = self.down1(x)
    x, skip2 = self.down2(x)
    x, skip3 = self.down3(x)
    x = self.bottleneck(x)

    x = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)(x)
    x = torch.cat([skip3, x], dim=1)
    x = nn.Conv1d(256, 128, kernel_size=3, padding=1)(x)

    x = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)(x)
    x = torch.cat([skip2, x], dim=1)
    x = nn.Conv1d(128, 64, kernel_size=3, padding=1)(x)

    x = nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2)(x)
    x = torch.cat([skip1, x], dim=1)
    x = nn.Conv1d(64, 32, kernel_size=3, padding=1)(x)

    x = self.final_layer(x)

    return x

model = UNet_Autoencoder(1, 1)
input = torch.randn(1, 1, 4096)
output = model(input)
print(output.shape)

# !unzip /content/train.zip

# Preparing the training dataset
from pathlib import Path
import datasets
from torch.utils.data import Dataset, DataLoader
from torchcodec.decoders import AudioDecoder

class AudioDataset(Dataset):
    def __init__(self, data_location):
        self.data_location = Path(data_location)
        self.bad_files = sorted(self.data_location.glob("*noised.wav"))
    def __len__(self):
        return len(self.bad_files)
    def __getitem__ (self, idx):
        bad_file = self.bad_files[idx]
        good_file = Path(str(bad_file).replace("-noised.wav", ".wav"))

        bad_decoder = AudioDecoder(bad_file)
        good_decoder = AudioDecoder(good_file)
        SLICE_LENGTH = 16384
        slice_seconds = SLICE_LENGTH / bad_decoder.metadata.sample_rate

        start_second = np.random.uniform(0, bad_decoder.metadata.duration_seconds - slice_seconds)
        waveform_bad = bad_decoder.get_samples_played_in_range(
            start_second,
            start_second + slice_seconds
        ).data

        waveform_good = good_decoder.get_samples_played_in_range(
            start_second,
            start_second + slice_seconds
        ).data

        return waveform_bad, waveform_good

# Training the model
# prepare all the cool things our model needs: optimizers, etc.
pink_songs = AudioDataset("/content/train")
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = torch.utils.data.DataLoader(pink_songs)
model.to(device)

def train_one_epoch(model, loader, optimizer, loss, device):
    running_loss = 0.0
    model.train()

    for i, (bad_audio, good_audio) in enumerate(loader):
        num_channels = bad_audio.shape[1]
        for j in range(num_channels):
            current_bad_audio = bad_audio[:, j:j+1, :].to(device)
            current_good_audio = good_audio[:, j:j+1, :].to(device)

            output = model(current_bad_audio)
            loss_val = loss(output, current_good_audio)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            running_loss += loss_val.item()

    return (running_loss / (len(loader) * num_channels))

num_epochs = 20
for epoch_id in range(num_epochs):
    loss_value = train_one_epoch(model, loader, optimizer, loss_fn, device)
    print(f"Loss in epoch {epoch_id}: {loss_value}")
# THIS WORKS, maybe tune hyperparameters now

# here, we write the test logic
# it's different, as we need to crack every chunk of length 1024, IN SEQUENCE

from torchcodec.encoders import AudioEncoder
def denoise_song(path, CHUNK_SIZE):
  decoder = AudioDecoder(path)
  song_tensor = decoder.get_all_samples().data
  sr = decoder.metadata.sample_rate

  length = song_tensor.shape[1]

  num_chunks = length // CHUNK_SIZE
  num_channels = song_tensor.shape[0]

  new_song_tensor = torch.zeros(num_channels, length)
  for j in range(num_channels):
     for i in range(num_chunks):
       curr_chunk = song_tensor[j:j+1, i*CHUNK_SIZE:(i+1)*CHUNK_SIZE]
       curr_chunk = curr_chunk.unsqueeze(1)
       output = model(curr_chunk)
       new_song_tensor[j, i*CHUNK_SIZE:(i+1)*CHUNK_SIZE] = output.squeeze(1)
  encoder = AudioEncoder(samples = new_song_tensor, sample_rate=sr)
  save_path = Path(str(path).replace(".wav", "-denoised.wav"))
  encoder.to_file(save_path)
# we run one small example, but it's the same for all .wav files
denoise_song("/content/train/arabic-noised.wav", 16384)
