"""
Audio File base
"""
import struct


class AudioFile:
    file = None

    def __init__(self, file: str):
        self.file = file
        print(self)

    def read_props(self) -> (int, int, int):
        """
        Reads audio file and return its number of channels,
        sample_rate and bit_depth.
        """
        audio_file = open(self.file, "rb")

        riff = audio_file.read(12)
        fmt = audio_file.read(36)

        num_channels_string = fmt[10:12]
        num_channels = struct.unpack('<H', num_channels_string)[0]

        sample_rate_string = fmt[12:16]
        sample_rate = struct.unpack("<I", sample_rate_string)[0]

        bit_depth_string = fmt[22:24]
        bit_depth = struct.unpack("<H", bit_depth_string)[0]

        return (num_channels, sample_rate, bit_depth)
