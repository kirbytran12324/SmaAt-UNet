import tifffile
import h5py
import os
import random
from datetime import datetime

# Directory containing the TIFF files
tiff_dir = '../dataset'

# Output HDF5 file
hdf5_file = '../dataset/composed/test.h5'


def load_tiff_files(directory):
    images = {}
    timestamps = []

    def search_for_tiff_files(directory):
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.tif'):
                    # Load the image
                    img = tifffile.imread(os.path.join(root, file))

                    # Extract the timestamp from the filename
                    timestamp_str = file[6:-4]
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")

                    # Store the image in the dictionary with the timestamp as the key
                    images[timestamp] = img
                    timestamps.append(timestamp)

        for root, dirs, _ in os.walk(directory):
            for d in dirs:
                search_for_tiff_files(os.path.join(root, d))  # Recursively search in subdirectories

    search_for_tiff_files(directory)
    return list(images.values()), timestamps


def save_to_hdf5(group_name, images, timestamps):
    with h5py.File(hdf5_file, 'a') as hf:
        group = hf.create_group(group_name)
        group.create_dataset('images', data=images)
        group.create_dataset('timestamps', data=timestamps)


def process_directory(directory):
    # Load TIFF images and timestamps
    images, timestamps = load_tiff_files(directory)

    # Initialize lists to hold training and testing data
    train_images, train_timestamps = [], []
    test_images, test_timestamps = [], []

    # Generate 7 random days for testing
    random_days = random.sample(range(1, 32), 7)

    # Iterate over images and timestamps
    for img, timestamp in zip(images, timestamps):
        # If the image is from October
        if timestamp.month == 10:
            # If the day of the month is one of the randomly selected days
            if timestamp.day in random_days:
                # Add to testing set
                test_images.append(img)
                test_timestamps.append(timestamp.strftime("%Y/%m/%d, %H:%M:%S"))
            else:
                # Otherwise, add to training set
                train_images.append(img)
                train_timestamps.append(timestamp.strftime("%Y/%m/%d, %H:%M:%S"))
        else:
            # If the image is not from October, add to training set
            train_images.append(img)
            train_timestamps.append(timestamp.strftime("%Y/%m/%d, %H:%M:%S"))

    # Save training and testing sets in HDF5 file
    save_to_hdf5('train', train_images, train_timestamps)
    save_to_hdf5('test', test_images, test_timestamps)


if __name__ == '__main__':
    process_directory(tiff_dir)
    print('Data saved to HDF5 file.')