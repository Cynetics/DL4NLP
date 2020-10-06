from .wili import *

def get_wili_data(config):
    """
    get train and validation data loaders.
    """

    data_path = config.data_path
    label_path = config.label_path

    val_data_path = config.val_data_path
    val_label_path = config.val_label_path

    train_data = WiliDataLoader(data_path=data_path,
                          label_path=label_path,
                          sequence_length=config.sequence_length
                         )

    val_data = WiliDataLoader(data_path=val_data_path,
                          label_path=val_label_path,
                          sequence_length=config.sequence_length
                         )

    return train_data, val_data

def get_wili_data_bytes(config):
    """
    get train and validation data loaders for bytes
    """

    data_path = config.data_path
    label_path = config.label_path

    val_data_path = config.val_data_path
    val_label_path = config.val_label_path

    train_data = WiliBytesDataLoader(data_path=data_path,
                          label_path=label_path,
                          sequence_length=config.sequence_length,
                         )

    val_data = WiliBytesDataLoader(data_path=val_data_path,
                          label_path=val_label_path,
                          sequence_length=config.sequence_length,
                        )
    return train_data, val_data
