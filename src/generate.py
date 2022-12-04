import argparse
from src.utils import get_device, get_the_latest_models, midi_to_wav, array_to_midi, generate_random_midi_array


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Music generation using GAN.')
    parser.add_argument('--seq_len', default=3, help='How many bars to connect? Bar is the basic building block.')
    parser.add_argument('--resolution', default=4, help='The bigger the resolution the slower the music.')
    parser.add_argument('--threshold', default=0.5, help='TODO')
    parser.add_argument('--name', help='Name of the file.')
    args = parser.parse_args()

    seq_len = args.seq_len
    resolution = args.resolution
    threshold = args.threshold
    name = args.name
    generator, _ = get_the_latest_models()

    device = get_device()
    generator = generator.to(device)
    generator.vectors_generator.sequence_length = seq_len

    img = generate_random_midi_array(generator=generator, device=device, threshold=threshold)
    array_to_midi(img, f'./{name}.midi', resolution=resolution)
    midi_to_wav(f'./{name}.midi', f'./{name}.wav')


