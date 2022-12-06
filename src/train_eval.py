import torch
import pandas as pd
import numpy as np
import pypianoroll
import muspy
from typing import List, Tuple
from src.constants import LATENT_DIM, NUMBER_OF_PITCHES, LOWEST_PITCH
from src.utils import write_model_params_to_tensorboard
from src.models import Discriminator, Generator
import torchvision.utils as vutils
from statistics import mean


class Metrics:
    """
    A class that determines metrics such as pitch_range.
    :param bar_generator: sequence generator model
    :param resolution: the bigger this value, the faster the music is
    :param threshold: decision threshold that tells whether to convert the value to 1 or 0
    :param probe: on how many data samples will be averaged metrics results
    :param device: device on which operations are made (cpu or gpu)
    """
    def __init__(self, bar_generator: Generator, resolution: float, threshold: float, probe: int, device: torch.device):
        self.bar_generator = bar_generator
        self.resolution = resolution
        self.threshold = threshold
        self.probe = probe
        self.device = device

    def generate_random_midi_array(self) -> np.ndarray:
        """
        Method that generates random bar array using a generator model.
        """
        img = self.bar_generator(torch.randn(1, 1, LATENT_DIM, device=self.device)).reshape(-1, NUMBER_OF_PITCHES).detach().cpu()
        img = img.numpy()
        img = img > self.threshold
        img = np.pad(img, ((0, 0), (LOWEST_PITCH, 128 - LOWEST_PITCH - NUMBER_OF_PITCHES)))
        return img.transpose()

    def array_to_pianoroll(self, music_array: np.ndarray) -> pypianoroll.multitrack.Multitrack:
        """
        Method that converts array to pianoroll Multitrack object.
        """
        pr = pypianoroll.Multitrack()
        pr.append(pypianoroll.BinaryTrack(pianoroll=music_array.transpose()))
        pr.resolution = self.resolution
        return pr

    def get_avg_metrics(self) -> List[float]:
        """
        Here we calculate metrics and study quality of the generator. We also average the results, that's why they are more informative. 
        """
        all_metrics = [[] for _ in range(10)]

        for _ in range(self.probe):
            array = self.generate_random_midi_array()
            pr = self.array_to_pianoroll(array)
            music = muspy.from_object(pr)

            all_metrics[0].append(muspy.pitch_range(music=music))
            all_metrics[1].append(muspy.n_pitches_used(music=music))
            all_metrics[2].append(muspy.polyphony(music=music))
            all_metrics[3].append(muspy.polyphony_rate(music=music))
            all_metrics[4].append(muspy.empty_beat_rate(music=music))
            all_metrics[5].append(muspy.empty_measure_rate(music=music, measure_resolution=1))
            all_metrics[6].append(muspy.groove_consistency(music=music, measure_resolution=1))
            all_metrics[7].append(muspy.pitch_class_entropy(music=music))
            all_metrics[8].append(muspy.pitch_entropy(music=music))
            all_metrics[9].append(muspy.scale_consistency(music=music))
        return [sum(metric)/len(metric) for metric in all_metrics]


    def create_metrics_df(self) -> pd.DataFrame:
        """
        Method that returns pandas dataframe with metrics information.
        """
        columns = ["pitch_range", "n_pitches_used", "polyphony", "polyphony_rate", "empty_beat_rate", "empty_measure_rate",
            "groove_consistency", "pitch_class_entropy", "pitch_entropy", "scale_consistency"]
        return pd.DataFrame(columns=columns, data=[self.get_avg_metrics()])


def compute_gradient_penalty(discriminator: Discriminator, real_samples: np.ndarray, fake_samples: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Method that computes gradient penalty. Essential function for WGAN.
    :param discriminator: discriminator model
    :param real_samples: sample of real images
    :param fake_samples: sample of fake images
    :param device: device on which operations are made (cpu or gpu)
    """
    # Get random interpolations between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples))
    interpolates = interpolates.requires_grad_(True)
    # Get the discriminator output for the interpolations
    d_interpolates = discriminator(interpolates)
    # Get gradients w.r.t. the interpolations
    fake = torch.ones(real_samples.size(0), 1).to(device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    # Compute gradient penalty
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# TODO
def train_one_step(discriminator_optimizer, generator_optimizer, discriminator: Discriminator,\
     generator: Generator, real_samples: np.ndarray, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Method that trains discriminator and generator models (just one step)
    :param discriminator_optimizer: discriminator optimizer
    :param generator_optimizer: generator optimizer
    :param discriminator: discriminator (critic) model
    :param generator: generator model
    :param real_samples: real samples of data to train
    :param device: device on which operations are made (cpu or gpu)
    :return discriminator_loss, generator_loss: losses after one step of training
    """
    # latent_vector = torch.randn(real_samples.shape[0], LATENT_DIM, 1, 1)
    latent_vector = torch.randn(real_samples.shape[0], 1, LATENT_DIM, device=device)

    real_samples = real_samples.to(device)
    for _ in range(5):
        # --> Train the discriminator (critic) <--
        discriminator_optimizer.zero_grad()
        prediction_real = discriminator(real_samples)
        
        fake_samples = generator(latent_vector)
        prediction_fake_discriminator = discriminator(fake_samples.detach())

        # Compute gradient penalty
        discriminator_loss = -(torch.mean(prediction_real) - torch.mean(prediction_fake_discriminator)) + \
            10.0 * compute_gradient_penalty(discriminator, real_samples.data, fake_samples.data, device=device)
        discriminator_loss.backward(retain_graph=True)

        # Update the weights
        discriminator_optimizer.step()
    
    # --> Train the generator <--
    generator_optimizer.zero_grad()

    prediction_fake_generator = discriminator(fake_samples)
    generator_loss = -torch.mean(prediction_fake_generator)
    generator_loss.backward()
    generator_optimizer.step()

    return discriminator_loss, generator_loss