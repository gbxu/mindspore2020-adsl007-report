from mindspore.communication import init
from mindspore.train.serialization import load_checkpoint
from mindspore.train.model import ParallelMode
from mindspore import Model
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore import context
import os, argparse, random, time
import numpy as np
import moxing as mox
from inceptionv4 import Inceptionv4
from dataset_imagenet import create_dataset, device_id, device_num

DATA_PATH = "obs://public-obs2020/pytorch-imagenet/val/"
CKPT_ROOT = "obs://mindspore-res-commit-adsl/" # directory of checkpoint files
META_FILE = ["inceptionv4-adsl007-final-version.meta"]  # meta file of the checkpoint to be loaded
CKPT_FILE = ["inceptionv4-adsl007-final-version.ckpt"]  # ckpt file of the checkpoint to be loaded

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

random.seed(1)
np.random.seed(1)

init(backend_name='hccl')

def Inceptionv4_eval():
    batch_size = 32
    class_num = 1000
    local_data_path = '/cache/data'

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(device_id=device_id)

    if device_num > 1:
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          mirror_mean=True)

    for meta_file in META_FILE:
        mox.file.copy(src_url=CKPT_ROOT+meta_file,
                      dst_url=os.path.join(local_data_path, meta_file))
    mox.file.copy_parallel(src_url=DATA_PATH, dst_url=local_data_path+"/val/")

    # create dataset
    eval_dataset = create_dataset(dataset_path=local_data_path, do_train=False,
                                  repeat_num=1, batch_size=batch_size)

    for ckpt_file in CKPT_FILE:
        if device_num == 1 or device_id == 0:
            print(ckpt_file, "\n")

            # create model
            net = Inceptionv4(classes=class_num, is_train=False)
            # load checkpoint
            mox.file.copy(src_url=CKPT_ROOT+ckpt_file,
                          dst_url=os.path.join(local_data_path, ckpt_file))
            load_checkpoint(os.path.join(local_data_path, ckpt_file), net=net)

            loss = SoftmaxCrossEntropyWithLogits(sparse=True)
            model = Model(net, loss_fn=loss, metrics={
                          'acc', 'top_1_accuracy', 'top_5_accuracy'})

            output = model.eval(eval_dataset, dataset_sink_mode=False)
            print(f'Evaluation result: {output["acc"]}.')
            print(f'TOP1: {output["top_1_accuracy"]}.')
            print(f'TOP5: {output["top_5_accuracy"]}.')


if __name__ == '__main__':
    Inceptionv4_eval()
    print('Inceptionv4 eval success!')
