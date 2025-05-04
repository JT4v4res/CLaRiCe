import tensorflow as tf
from huggingface_hub import snapshot_download
import os


def load_model():
    local_dir = snapshot_download(repo_id='JT4v4res/CLaRiCe', token=None)

    # print(os.path.join(local_dir, 'CLaRiCe'))

    clarice = tf.saved_model.load(os.path.join(local_dir, 'CLaRiCe'))

    return clarice.signatures["serving_default"]
