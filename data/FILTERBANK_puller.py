"""

This script downloads filterbank data from SETI Berkeley Open Data API.

Usage:
```
python data/FILTERBANK_puller.py --targets "HIP17147" --telescope "GBT" --save-dir ".data/BLIS692NS/BLIS692NS_data"
```

This will download Breakthrough Listening Search 692 Nearby Stars (BLIS692NS) filterbank data for HIP17147 from the GBT telescope.

You can also use the script to download other types of files by changing the `--file-type` argument.

Note: The script uses the SETI Berkeley Open Data API, which may be subject to change.
"""
import argparse
import os

import requests
from tqdm import tqdm


def download_file(url, save_dir):
    """Download a file with progress bar"""
    filename = os.path.basename(url)
    save_path = os.path.join(save_dir, filename)

    try:
        # Head request to get file size for progress bar
        response = requests.head(url)
        file_size = int(response.headers.get('content-length', 0))

        # Start download with progress bar
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            progress_bar = tqdm(
                desc=filename[:20],  # Truncate long names
                total=file_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            )

            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    progress_bar.update(len(chunk))

            progress_bar.close()
            print(f"Downloaded {filename} to {save_path}")

    except requests.RequestException as e:
        print(f"Failed to download {filename}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Download filterbank data from SETI Berkeley open data')
    parser.add_argument('--targets', required=True, help='Comma-separated list of target IDs')
    parser.add_argument('--save-dir', default='.', help='Directory to save the downloaded files')
    parser.add_argument('--file-type', default='filterbank', help='Type of files to download')
    parser.add_argument('--telescope', help='Filter by telescope')
    parser.add_argument('--grade', help='Filter by grade')
    parser.add_argument('--quality', help='Filter by quality')
    parser.add_argument('--limit', type=int, default=10, help='Maximum number of files to return per target')
    parser.add_argument('--time-start', type=float, help='Start time in MJD for filtering observations')
    parser.add_argument('--time-end', type=float, help='End time in MJD for filtering observations')
    parser.add_argument('--freq-start', type=float, help='Start frequency in MHz for center frequency filtering')
    parser.add_argument('--freq-end', type=float, help='End frequency in MHz for center frequency filtering')

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    targets = args.targets.split(',')

    print(f"Starting download of {len(targets)} targets to {args.save_dir}")
    print(f"File type: {args.file_type}, Limit: {args.limit} files per target")

    for target in targets:
        params = {
            'target': target,
            'file-types': args.file_type,
            'limit': args.limit
        }
        if args.telescope:
            params['telescopes'] = args.telescope
        if args.grade:
            params['grades'] = args.grade
        if args.quality:
            params['quality'] = args.quality
        if args.time_start is not None:
            params['time-start'] = args.time_start
        if args.time_end is not None:
            params['time-end'] = args.time_end
        if args.freq_start is not None:
            params['freq-start'] = args.freq_start
        if args.freq_end is not None:
            params['freq-end'] = args.freq_end

        url = 'http://seti.berkeley.edu/opendata/api/query-files'
        try:
            print(f"\nQuerying files for target: {target}")
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if data['result'] == 'success':
                    if not data['data']:
                        print(f"  No files found for target {target}")
                        continue

                    file_count = len(data['data'])
                    print(f"  Found {file_count} files, starting download...")

                    for entry in tqdm(data['data'], desc=f"Downloading {target}", unit='file'):
                        if entry['target'] == target:  # exact match
                            file_url = entry['url']
                            download_file(file_url, args.save_dir)
                else:
                    print(f"Error querying target {target}: {data['message']}")
            else:
                print(f"Failed to query target {target}: HTTP {response.status_code}")
        except requests.RequestException as e:
            print(f"Failed to query target {target}: {e}")

    print("\nAll downloads completed!")


if __name__ == '__main__':
    """
    Example usage:
    HIP17147 03/11/2017 07:22:49 6001.4648(1.8G)：
    python data/FILTERBANK_puller.py --targets "HIP17147" --telescope "GBT" --save-dir "./data/BLIS692NS/BLIS692NS_data" --time-start 58060.3074 --time-end 58060.3076 --freq-start 6001.4633 --freq-end 6001.4635
    HIP17147 03/11/2017 07:22:49 ALL:
    python data/FILTERBANK_puller.py --targets "HIP17147" --telescope "GBT" --save-dir "./data/BLIS692NS_data" --time-start 58060.3074 --time-end 58060.3076
    """
    main()
