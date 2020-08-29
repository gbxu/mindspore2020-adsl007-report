"""Create train or eval dataset."""
import os
import mindspore.common.dtype as mstype
import mindspore.dataset as de
import mindspore.dataset.transforms.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2


device_id = int(os.getenv('DEVICE_ID'))
device_num = int(os.getenv('RANK_SIZE'))


def create_dataset(dataset_path, do_train, repeat_num=1, batch_size=32):
    """
    Create a train or eval dataset.

    Args:
        dataset_path (str): The path of dataset.
        do_train (bool): Whether dataset is used for train or eval.
        repeat_num (int): The repeat times of dataset. Default: 1.
        batch_size (int): The batch size of dataset. Default: 32.

    Returns:
        Dataset.
    """

    if do_train:
        dataset_path = os.path.join(dataset_path, 'train')
        do_shuffle = True
    else:
        dataset_path = os.path.join(dataset_path, 'val')
        do_shuffle = False

    if device_num == 1 or not do_train:
        ds = de.ImageFolderDatasetV2(dataset_path, decode=True, extensions=[".JPEG"], num_parallel_workers=192, 
                                    shuffle=do_shuffle)
    else:

        ds = de.ImageFolderDatasetV2(dataset_path, decode=True, extensions=[".JPEG"], num_parallel_workers=192, 
                                    shuffle=do_shuffle, num_shards=device_num, shard_id=device_id)

    buffer_size = 50
    rescale = 1.0 / 255.0
    shift = 0.0    

    # define map operations
    normalize_op = C.Normalize([123.6, 116.3, 103.5], [58.4, 57.1, 57.4])
    rescale_op = C.Rescale(rescale, shift)    

    trans = [rescale_op, normalize_op]
    type_cast_op = C2.TypeCast(mstype.int32)

    ds = ds.map(input_columns="label", operations=type_cast_op)
    ds = ds.map(input_columns="image", operations=trans)

    # apply shuffle operations
    ds = ds.shuffle(buffer_size=buffer_size)

    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)

    # apply dataset repeat operation
    ds = ds.repeat(repeat_num)

    return ds