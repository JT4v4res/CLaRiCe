import tensorflow as tf


def load_model():
    clarice = tf.saved_model.load('./pred/model/conv1d_7_study_3/conv1d_7_study_3/')

    return clarice.signatures["serving_default"]
