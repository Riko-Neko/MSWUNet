import os
import sys
import time

import numpy as np
import torch

SHOWING_BACKEND = False

from blimpy.io import file_wrapper as fw

from astropy.time import Time
from astropy import units as u

# Logging set up
import logging

logger = logging.getLogger(__name__)

level_log = logging.INFO

if level_log == logging.INFO:
    stream = sys.stdout
    fmt = '%(name)-15s %(levelname)-8s %(message)s'
else:
    stream = sys.stderr
    fmt = '%%(relativeCreated)5d (name)-15s %(levelname)-8s %(message)s'

logging.basicConfig(format=fmt, stream=stream, level=level_log)

MAX_BLOB_MB = 1024


class Waterfall():
    """ Class for loading and writing blimpy data (.fil, .h5) """

    def __repr__(self):
        return "Waterfall data: %s" % self.filename

    def _init_alternate(self, header_dict, data, filename=None):
        pass

    def __init__(self, filename=None, f_start=None, f_stop=None, t_start=None, t_stop=None,
                 load_data=True, max_load=None, header_dict=None, data_array=None):
        """ Class for loading and plotting blimpy data.

        This class parses the blimpy file and stores the header and data
        as objects:
            fb = Waterfall('filename_here.fil')
            fb.header        # blimpy header, as a dictionary
            fb.data          # blimpy data, as a numpy array

        Args:
            filename (str): filename of blimpy file.  REQUIRED.
            f_start (float): start frequency in MHz
            f_stop (float): stop frequency in MHz
            t_start (int): start integration ID
            t_stop (int): stop integration ID
            load_data (bool): load data. If set to False, only header will be read.
            max_load (float): maximum data to load in GB.
            header_dict (dict): *NOT CURRENTLY SUPPORTED*
            data_array (np.array): *NOT CURRENTLY SUPPORTED*
        """

        if (header_dict is not None) or (data_array is not None):
            self._init_alternate(header_dict, data_array, filename=filename)
            return

        if filename is None:
            raise ValueError("Currently, a value for filename must be supplied.")

        self.filename = filename
        self.ext = os.path.splitext(filename)[-1].lower()
        self.container = fw.open_file(filename, f_start=f_start, f_stop=f_stop, t_start=t_start, t_stop=t_stop,
                                      load_data=load_data, max_load=max_load)
        self.file_header = self.container.header
        self.header = self.file_header
        self.n_ints_in_file = self.container.n_ints_in_file
        self.file_shape = self.container.file_shape
        self.file_size_bytes = self.container.file_size_bytes
        self.selection_shape = self.container.selection_shape
        self.n_channels_in_file = self.container.n_channels_in_file

        # Cache freqs and timestamps once in init
        start = time.time()
        self.freqs = self.container.populate_freqs()  # Cache freqs
        print(f"Waterfall.__init__ - populate_freqs: {time.time() - start:.3f} seconds")
        start = time.time()
        self.timestamps = self.container.populate_timestamps()  # Cache timestamps
        print(f"Waterfall.__init__ - populate_timestamps: {time.time() - start:.3f} seconds")

        start = time.time()
        self.is_monotonic_inc = np.all(np.diff(self.freqs) >= 0)
        self.is_monotonic_dec = np.all(np.diff(self.freqs) <= 0)
        print(f"Waterfall.__init__ - is_monotonic_inc/dec: {time.time() - start:.3f} seconds")

        # These values will be modified once code for multi_beam and multi_stokes observations are possible.
        self.freq_axis = 2
        self.time_axis = 0
        self.beam_axis = 1  # Place holder # Polarisation?
        self.stokes_axis = 4  # Place holder

        self.logger = logger

        self.__load_data()

    def __load_data(self):
        """ Helper for loading data from a container. Should not be called manually. """

        self.data = self.container.data

    def get_freqs(self):
        """
        Get the frequency array for this Waterfall object.

        Returns
        -------
        numpy array
            Values for all of the fine frequency channels.

        """
        return np.arange(0, self.header['nchans'], 1, dtype=float) \
            * self.header['foff'] + self.header['fch1']

    def read_data(self, f_start=None, f_stop=None, t_start=None, t_stop=None):
        """ Reads data selection if small enough.

        Args:
            f_start (float): Start frequency in MHz
            f_stop (float): Stop frequency in MHz
            t_start (int): Integer time index to start at
            t_stop (int): Integer time index to stop at

        Data is loaded into self.data (nothing is returned)
        """

        self.container.read_data(f_start=f_start, f_stop=f_stop, t_start=t_start, t_stop=t_stop)

        self.__load_data()

    def _update_header(self):
        """ Updates the header information from the original file to the selection. """

        # Updating frequency of first channel from selection
        if self.header['foff'] < 0:
            self.header['fch1'] = self.container.f_stop
        else:
            self.header['fch1'] = self.container.f_start

        # Updating number of fine channels.
        self.header['nchans'] = self.container.selection_shape[self.freq_axis]

        # Updating time stamp for first time bin from selection
        self.header['tstart'] = self.container.populate_timestamps(update_header=True)

    def info(self):
        """ Print header information and other derived information. """

        print("\n--- File Info ---")

        for key, val in self.file_header.items():
            if key == 'src_raj':
                val = val.to_string(unit=u.hour, sep=':')
            if key == 'src_dej':
                val = val.to_string(unit=u.deg, sep=':')
            if key in ('foff', 'fch1'):
                val *= u.MHz
            if key == 'tstart':
                print("%16s : %32s" % ("tstart (ISOT)", Time(val, format='mjd').isot))
                key = "tstart (MJD)"
            print("%16s : %32s" % (key, val))

        print("\n%16s : %32s" % ("Num ints in file", self.n_ints_in_file))
        print("%16s : %32s" % ("File shape", self.file_shape))
        print("--- Selection Info ---")
        print("%16s : %32s" % ("Data selection shape", self.selection_shape))
        if self.header['foff'] < 0:  # descending frequency values
            minfreq = self.container.f_start - self.header['foff']
            maxfreq = self.container.f_stop
        else:  # ascending frequency values
            minfreq = self.container.f_start
            maxfreq = self.container.f_stop - self.header['foff']
        print("%16s : %32s" % ("Minimum freq (MHz)", minfreq))
        print("%16s : %32s" % ("Maximum freq (MHz)", maxfreq))

    def _get_blob_dimensions(self, chunk_dim):
        """ Computes the blob dimensions, trying to read around 1024 MiB at a time.
            This is assuming a chunk is about 1 MiB.

            Notes:
                A 'blob' is the max size that will be read into memory at once.
                A 'chunk' is a HDF5 concept to do with efficient read access, see
                https://portal.hdfgroup.org/display/HDF5/Chunking+in+HDF5

            Args:
                chunk_dim (array of ints): Shape of chunk, e.g. (1024, 1, 768)

            Returns blob dimensions (shape = array of ints).
        """

        # Taking the size into consideration, but avoiding having multiple blobs within a single time bin.
        freq_axis_size = self.selection_shape[self.freq_axis]
        if self.selection_shape[self.freq_axis] > chunk_dim[self.freq_axis] * MAX_BLOB_MB:
            time_axis_size = 1
        else:
            time_axis_size = np.min(
                [chunk_dim[self.time_axis] * MAX_BLOB_MB * chunk_dim[self.freq_axis] / freq_axis_size,
                 self.selection_shape[self.time_axis]])

        blob_dim = (int(time_axis_size), 1, freq_axis_size)

        return blob_dim

    def _get_chunk_dimensions(self):
        """ Sets the chunking dimensions depending on the file type.

            Notes: A 'chunk' is a HDF5 concept to do with efficient read access, see
            https://portal.hdfgroup.org/display/HDF5/Chunking+in+HDF5

            Returns chunk dimensions (shape), e.g. (2048, 1, 512)
        """

        # Usually '.0000.' is in self.filename
        if np.abs(self.header['foff']) < 1e-5:
            logger.info('Detecting high frequency resolution data.')
            chunk_dim = (1, 1, 1048576)  # 1048576 is the number of channels in a coarse channel.
            return chunk_dim

        # Usually '.0001.' is in self.filename
        if np.abs(self.header['tsamp']) < 1e-3:
            logger.info('Detecting high time resolution data.')
            chunk_dim = (2048, 1, 512)  # 512 is the total number of channels per single band (ie. blc00)
            return chunk_dim

        # Usually '.0002.' is in self.filename
        if np.abs(self.header['foff']) < 1e-2 and np.abs(self.header['foff']) >= 1e-5:
            logger.info('Detecting intermediate frequency and time resolution data.')
            chunk_dim = (10, 1, 65536)  # 65536 is the total number of channels per single band (ie. blc00)
            return chunk_dim

        logger.warning('File format not known. Will use minimum chunking. NOT OPTIMAL.')
        chunk_dim = (1, 1, 512)
        return chunk_dim

    def calc_n_coarse_chan(self, chan_bw=None):
        """ This makes an attempt to calculate the number of coarse channels in a given freq selection.

            Note:
                This is unlikely to work on non-Breakthrough Listen data, as a-priori knowledge of
                the digitizer system is required.

            Returns n_coarse_chan (int), number of coarse channels
        """

        n_coarse_chan = self.container.calc_n_coarse_chan(chan_bw)

        return n_coarse_chan

    def grab_data(self, f_start=None, f_stop=None, t_start=None, t_stop=None, if_id=0, verbose=False, device='cpu'):
        """ Extract a portion of data by frequency range.

        Args:
            f_start (float): start frequency in MHz
            f_stop (float): stop frequency in MHz
            t_start (int): start integration ID
            t_stop (int): stop integration ID
            if_id (int): IF input identification
            verbose (bool): Whether to print timing information

        Returns:
            (freqs, data) (np.arrays): frequency axis in MHz and data subset
        """
        start_total = time.time()

        # Step 1: Check if container is heavy
        start = time.time()
        if self.container.isheavy():
            raise Exception(
                "Waterfall.grab_data: Large data array was not loaded! Try instantiating Waterfall with max_load.")
        if verbose:
            print(f"Step 1 - isheavy check: {time.time() - start:.3f} seconds")

        # Step 2: Use cached frequencies
        start = time.time()
        if not hasattr(self, 'freqs') or self.freqs is None:
            self.freqs = self.container.populate_freqs()
        if verbose:
            print(f"Step 2 - populate_freqs (or cached): {time.time() - start:.3f} seconds")

        # Step 3: Use cached timestamps
        start = time.time()
        if not hasattr(self, 'timestamps') or self.timestamps is None:
            self.timestamps = self.container.populate_timestamps()
        if verbose:
            print(f"Step 3 - populate_timestamps (or cached): {time.time() - start:.3f} seconds")

        # Step 4: Set default frequency range
        start = time.time()
        if f_start is None:
            f_start = self.freqs[0]
        if f_stop is None:
            f_stop = self.freqs[-1]
        if verbose:
            print(f"Step 4 - Set default frequency range: {time.time() - start:.3f} seconds")

        # Step 5: Frequency indexing
        start = time.time()
        if verbose and (not self.is_monotonic_inc and not self.is_monotonic_dec):
            print(f"[\033[32mInfo\033[0m] Using {device} for data extraction.")
        elif verbose:
            print(f"[\033[32mInfo\033[0m] Using numpy.searchsorted for data extraction.")

        def nearest_index(freqs, f, idx):
            # 边界保护
            if idx == 0:
                return 0
            elif idx == len(freqs):
                return len(freqs) - 1
            # 检查左右哪个更近（相等时取左边，保持与 argmin 一致）
            left, right = freqs[idx - 1], freqs[idx]
            if abs(f - left) <= abs(f - right):
                return idx - 1
            return idx

        if self.is_monotonic_inc:
            i0 = nearest_index(self.freqs, f_start, np.searchsorted(self.freqs, f_start, side="left"))
            i1 = nearest_index(self.freqs, f_stop, np.searchsorted(self.freqs, f_stop, side="left"))
            path_used = "numpy.searchsorted (ascending + nearest)"
        elif self.is_monotonic_dec:
            rev_freqs = self.freqs[::-1]
            j0 = nearest_index(rev_freqs, f_start, np.searchsorted(rev_freqs, f_start, side="left"))
            j1 = nearest_index(rev_freqs, f_stop, np.searchsorted(rev_freqs, f_stop, side="left"))
            n = len(self.freqs)
            i0, i1 = n - 1 - j0, n - 1 - j1
            path_used = "numpy.searchsorted (descending + nearest)"
        else:
            if str(device).startswith("cuda"):
                freqs_t = torch.as_tensor(self.freqs, device=device, dtype=torch.float64)
                f0 = torch.tensor(float(f_start), device=device, dtype=torch.float64)
                f1 = torch.tensor(float(f_stop), device=device, dtype=torch.float64)
                i0 = int(torch.argmin(torch.abs(freqs_t - f0)).item())
                i1 = int(torch.argmin(torch.abs(freqs_t - f1)).item())
                path_used = f"torch.{device}"
            else:
                i0 = int(np.argmin(np.abs(self.freqs - f_start)))
                i1 = int(np.argmin(np.abs(self.freqs - f_stop)))
                path_used = "numpy.argmin"

        if verbose:
            print(f"Step 5 - Frequency indexing ({path_used}): {time.time() - start:.3f} seconds")

        # Step 6: Slice data
        start = time.time()
        if i0 < i1:
            plot_f = self.freqs[i0:i1 + 1]
            plot_data = np.squeeze(self.data[t_start:t_stop, if_id, i0:i1 + 1]).astype(np.float32)
        else:
            plot_f = self.freqs[i1:i0 + 1]
            plot_data = np.squeeze(self.data[t_start:t_stop, if_id, i1:i0 + 1]).astype(np.float32)
        if verbose:
            print(f"Step 6 - Data slicing and squeeze: {time.time() - start:.3f} seconds")

        # Print total time
        if verbose:
            print(f"Total grab_data execution: {time.time() - start_total:.3f} seconds")

        return plot_f, plot_data
