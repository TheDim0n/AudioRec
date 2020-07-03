import tensorflow as tf


if __name__=='__main__':
    GPU = tf.config.list_logical_devices('GPU')
    CPU = tf.config.list_logical_devices('CPU')
    DEVICE = GPU[0].name if GPU else CPU[0].name
    print("!Enabled device: ", DEVICE)