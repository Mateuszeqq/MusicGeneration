import pypianoroll
from PIL import Image
import numpy as np
from constants import LATENT_DIM
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from torch.utils.tensorboard import SummaryWriter

from constants import MUSIC_LENGTH, NUMBER_OF_PITCHES, LOWEST_PITCH


def write_model_params_to_tensorboard(tb_writer, model, epoch, prefix):
    for name, param in model.named_parameters():
        tb_writer.add_histogram(prefix + name, param, epoch)
        tb_writer.add_histogram(prefix + name + '_grad', param.grad, epoch)


def write_models_architecture_to_tensorboard(generator, discriminator, real_images_sample, noise):
    # It's necessary to redefine writer (idk why)
    writer = SummaryWriter()
    writer.add_graph(generator, noise)
    writer.close()

    writer = SummaryWriter()
    writer.add_graph(discriminator, real_images_sample)
    writer.close()


def binarize_array(array: np.ndarray, threshold: float) -> np.ndarray:
    array[array >= threshold] = 1
    array[array < threshold] = 0
    array = array.astype(np.uint8)
    return array


def generate_random_midi_array(generator, device, threshold: float) -> np.ndarray:
    img = generator(torch.randn(1, 1, LATENT_DIM, device= device)).reshape(-1, NUMBER_OF_PITCHES).detach().cpu()
    img = img.numpy()
    img = img > threshold
    img = np.pad(img, ((0, 0), (LOWEST_PITCH, 128 - LOWEST_PITCH - NUMBER_OF_PITCHES)))
    return img.transpose()


def array_to_midi(music_array, midi_path, plot=False, resolution=6):
    pr = pypianoroll.Multitrack()
    pr.append(pypianoroll.BinaryTrack(pianoroll=music_array.transpose()))
    pr.resolution = resolution
    if plot:
        pr.plot()
    pr.write(midi_path)
    return pr


def show_learning_process(img_list, threshold=0.5):
    img_list = [img.reshape(MUSIC_LENGTH, NUMBER_OF_PITCHES) > threshold for img in img_list]
    img_list = [img.astype(int) * 255 for img in img_list]
    img_list = [np.pad(img, ((0, 0), (LOWEST_PITCH, 128 - LOWEST_PITCH - NUMBER_OF_PITCHES))) for img in img_list]
    fig = plt.figure(figsize=(12, 12))
    plt.axis("off")
    ims = [[plt.imshow(img.transpose(), cmap='gray', animated=True)] for img in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=250, repeat_delay=250, blit=True)

    return HTML(ani.to_jshtml())


def get_pitch_range_tuple(img_path):
    img = Image.open(img_path)
    array = np.array(img)
    pr = pypianoroll.Multitrack()
    pr.append(pypianoroll.BinaryTrack(pianoroll=array.transpose()))
    return pypianoroll.pitch_range_tuple(pr.tracks[0].pianoroll)


def get_device():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    return device


def save_models(discriminator, generator, prefix):
    torch.save(generator, f'generators/{prefix}_generator.pt')
    torch.save(discriminator, f'discriminators/{prefix}_discriminator.pt')


def write_samples(name, generator, device, threshold, resolution=4):
    img = generate_random_midi_array(generator=generator, device=device, threshold=threshold)
    plt.imsave(fname=f'samples/images/{name}.png', arr=img, cmap='gray')
    array_to_midi(img, f'samples/music/{name}.midi', resolution=resolution)


def write_losses_to_tensorboard(writer, critic_loss, generator_loss, step):
    writer.add_scalar('critic_loss/loss', critic_loss, step)
    writer.add_scalar('generator_loss/loss', generator_loss, step)
