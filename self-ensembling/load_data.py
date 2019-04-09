"""
Data loading functions
"""
import tensorflow as tf

def dataset_download(files_to_download, url, train_index=0, test_index=1):
    """
    Download url/file for file in files_to_download
    Returns: the downloaded filename at train_index, test_index (e.g. 0 and 1,
        if you passed the train filename first and test filename second).
    """
    downloaded_files = []
    for f in files_to_download:
        downloaded_files.append(tf.keras.utils.get_file(fname=f, origin=url+"/"+f))
    train_fp = downloaded_files[train_index]
    test_fp = downloaded_files[test_index]
    return train_fp, test_fp

def tf_dataset(data, labels, buffer_size=60000, batch_size=256,
        prefetch_buffer_size=1, eval_shuffle_seed=0):
    return tf.data.Dataset.from_tensor_slices(data, labels).\
        shuffle(buffer_size, seed=eval_shuffle_seed).batch(batch_size).\
        prefetch(prefetch_buffer_size)

def load_dataset(train_images, train_labels, test_images, test_labels):
    """
    Load the X dataset as a tf.data.Dataset from train/test images/labels

    Returns: train_dataset, test_dataset

    Normal usage, for example:
        usps_train, usps_test = load_dataset(*load_usps())
    """
    train_dataset = tf_dataset(train_images, train_labels)
    test_dataset = tf_dataset(test_images, test_labels)
    return train_dataset, test_dataset
