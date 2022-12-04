import pypianoroll
import muspy
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from torch.utils.tensorboard import SummaryWriter
from src.constants import MUSIC_LENGTH, NUMBER_OF_PITCHES, LOWEST_PITCH, LATENT_DIM
from src.models import TemporalVectors, SequenceBarGenerator, GeneratorBlock, Generator, LayerNorm, DiscriminatorBlock, Discriminator
import glob
import os


def get_the_latest_models():
    """
    Method that returns the latest generator and discriminator.
    :return: generator and discriminator models
    """
    list_of_generators = glob.glob('models/generators/*')
    list_of_discriminators = glob.glob('models/discriminators/*')
    latest_generator = max(list_of_generators, key=os.path.getctime)
    latest_discriminator = max(list_of_discriminators, key=os.path.getctime)
    generator = torch.load(latest_generator)
    discriminator = torch.load(latest_discriminator)
    return generator, discriminator


def midi_to_wav(midi_path: str, output_path: str) -> None:
    """
    Method that converts midi file to wav using muspy library.
    :param str midi_path: File path to midi file
    :param str midi_path: File path to desired wav file
    """
    music = muspy.read(midi_path)
    muspy.outputs.write_audio(path=output_path, music=music)


def write_model_params_to_tensorboard(tb_writer, model, epoch, prefix) -> None:
    """
    Method that writes given model parameters to a tensorboard.
    :param tb_writer: Tensorboard writer instance
    :param model: Which model parameters to write
    :param int epoch: Define an epoch for an x-axis in a plot
    :param str prefix: Prefix to a plot title
    """
    for name, param in model.named_parameters():
        tb_writer.add_histogram(prefix + name, param, epoch)
        tb_writer.add_histogram(prefix + name + '_grad', param.grad, epoch)


def write_models_architecture_to_tensorboard(generator, discriminator, real_images_sample, noise):
    """
    Method that writes given model parameters to a tensorboard.
    :param generator: generator model
    :param discriminator: discriminator model
    :param real_images_sample: Sample of real images for discriminator
    :param noise: Sample of latent vector (noise) for generator
    """
    # It's necessary to redefine writer (idk why)
    writer = SummaryWriter()
    writer.add_graph(generator, noise)
    writer.close()

    writer = SummaryWriter()
    writer.add_graph(discriminator, real_images_sample)
    writer.close()


def binarize_array(array: np.ndarray, threshold: float) -> np.ndarray:
    """
    Method that maps a given array to a binary array.
    :param np.ndarray array: an array to transform to binary
    :param float threshold: decision threshold that tells whether to convert the value to 1 or 0
    """
    array[array >= threshold] = 1
    array[array < threshold] = 0
    array = array.astype(np.uint8)
    return array


def generate_random_midi_array(generator, device, threshold: float) -> np.ndarray:
    """
    Method that generates a meaningful array out of a noise using generator model.
    :param generator: a generator model, that maps noise into a meaningful array
    :param device: device on which operations are made (cpu or gpu)
    :param threshold: decision threshold that tells whether to convert the value to 1 or 0
    """
    img = generator(torch.randn(1, 1, LATENT_DIM, device= device)).reshape(-1, NUMBER_OF_PITCHES).detach().cpu()
    img = img.numpy()
    img = img > threshold
    img = np.pad(img, ((0, 0), (LOWEST_PITCH, 128 - LOWEST_PITCH - NUMBER_OF_PITCHES)))
    return img.transpose()


def array_to_midi(music_array, midi_path, plot=False, resolution=6):
    """
    Method that converts an array to midi file.
    :param music_array: an array to be converted
    :param midi_path: where to save a midi file
    :param plot: show plot of the pianoroll or not
    :param resolution: the bigger this value, the faster the music is
    """
    pr = pypianoroll.Multitrack()
    pr.append(pypianoroll.BinaryTrack(pianoroll=music_array.transpose()))
    pr.resolution = resolution
    if plot:
        pr.plot()
    pr.write(midi_path)
    return pr


def show_learning_process(img_list, threshold=0.5):
    """
    Method that ilustrates the learning process.
    :param img_list: list of arrays to do a slideshow of the learning process
    :param threshold: decision threshold that tells whether to convert the value to 1 or 0
    """
    img_list = [img.reshape(MUSIC_LENGTH, NUMBER_OF_PITCHES) > threshold for img in img_list]
    img_list = [img.astype(int) * 255 for img in img_list]
    img_list = [np.pad(img, ((0, 0), (LOWEST_PITCH, 128 - LOWEST_PITCH - NUMBER_OF_PITCHES))) for img in img_list]
    fig = plt.figure(figsize=(12, 12))
    plt.axis("off")
    ims = [[plt.imshow(img.transpose(), cmap='gray', animated=True)] for img in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=250, repeat_delay=250, blit=True)

    return HTML(ani.to_jshtml())


def get_pitch_range_tuple(img_path):
    """
    Method that brings out the tuple range from a photo that represents music
    :param img_path: path to an image
    """
    img = Image.open(img_path)
    array = np.array(img)
    pr = pypianoroll.Multitrack()
    pr.append(pypianoroll.BinaryTrack(pianoroll=array.transpose()))
    return pypianoroll.pitch_range_tuple(pr.tracks[0].pianoroll)


def get_device():
    """
    Method that returns device. CUDA if it's avaible, CPU otherwise.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    return device


def save_models(discriminator, generator, prefix):
    """
    Method that saves model to a directory.
    :param discriminator: discriminator model
    :param generator: generator model
    :param prefix: prefix to distinguish by epochs
    """
    torch.save(generator, f'../models/generators/{prefix}_generator.pt')
    torch.save(discriminator, f'../models/discriminators/{prefix}_discriminator.pt')


def write_samples(name, generator, device, threshold, resolution=4):
    """
    Method that saves samples (images, midi and wav files).
    :param name: name of the file (image, midi, wav)
    :param generator: generator model that will create music
    :param device: device on which operations are made (cpu or gpu)
    :param threshold: decision threshold that tells whether to convert the value to 1 or 0
    :param resolution: the bigger this value, the faster the music is
    """
    img = generate_random_midi_array(generator=generator, device=device, threshold=threshold)
    plt.imsave(fname=f'../samples/images/{name}.png', arr=img, cmap='gray')
    array_to_midi(img, f'../samples/midi/{name}.midi', resolution=resolution)
    midi_to_wav(f'../samples/midi/{name}.midi', f'../samples/music/{name}.wav')


def write_losses_to_tensorboard(writer, critic_loss, generator_loss, step):
    """
    Method that writes losses (generator and discriminator) to the tensorboard.
    :param writer: tensorboard writer instance
    :param critic_loss: value of the critic loss
    :param generator_loss: value of the generator loss
    :param step: define a step for an x-axis in a plot in the tensorboard
    """
    writer.add_scalar('critic_loss/loss', critic_loss, step)
    writer.add_scalar('generator_loss/loss', generator_loss, step)
