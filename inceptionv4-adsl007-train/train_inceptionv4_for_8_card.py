from mindspore.communication import init
from mindspore.train.serialization import load_checkpoint
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.model import ParallelMode
from mindspore.train.loss_scale_manager import FixedLossScaleManager, DynamicLossScaleManager
from mindspore.train.callback import Callback
from mindspore import Model
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.nn import RMSProp
from mindspore import Tensor
from mindspore import context
import os
import argparse
import random
import time
import numpy as np
import moxing as mox
from inceptionv4 import Inceptionv4
from dataset_imagenet import create_dataset, device_id, device_num

MEASURE_PERFMANCE = True
EPOCH_SIZE = 20     # number of epochs to run
CKPT_PREFIX = "inceptionv4-adsl007-train-8cards"   # prefix of checkpoint file to be saved
CKPT_ROOT = "obs://mindspore-res-commit-adsl/"   # directory of checkpoint files for loading and saving

META_FILE = ""  # meta file of the checkpoint to be loaded
CKPT_FILE = ""  # ckpt file of the checkpoint to be loaded
USE_CKPT = False # enable to train from the checkpoint

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
random.seed(1)
np.random.seed(1)
init(backend_name='hccl')


class PerformanceCallback(Callback):
    """
    Training performance callback.

    Args:
        batch_size (int): Batch number for one step.
    """

    def __init__(self, batch_size):
        super(PerformanceCallback, self).__init__()
        self.batch_size = batch_size
        self.last_step = 0
        self.epoch_begin_time = 0
        self.start_time = 0
        self.avg = 1.1

    def epoch_begin(self, run_context):
        self.t1 = time.time()

    def epoch_end(self, run_context):
        params = run_context.original_args()
        self.t2 = time.time()
        cost_time = self.t2 - self.t1
        print(f'epoch {params.cur_epoch_num} cost time = {cost_time} s, '
                f'one step time: {1000*cost_time/(1281167/(128*8))} ms\n')

    def step_begin(self, run_context):
        if self.start_time == 0:
            self.start_time = time.time()
        self.epoch_begin_time = time.time()

    def step_end(self, run_context):
        params = run_context.original_args()
        self.epoch_end_time = time.time()
        cost_time = self.epoch_end_time - self.epoch_begin_time
        train_steps = params.cur_step_num - self.last_step
        self.avg = self.avg*0.99+cost_time*0.01

        if params.cur_step_num % 200 == 0 and device_id == 0:
            print(f'epoch {params.cur_epoch_num} cost time = {cost_time}, train step num: {train_steps}, '
                  f'one step time: {1000*cost_time/train_steps} ms, avg: {1000*self.avg/train_steps} ms '
                  f'train samples per second of cluster: {device_num*train_steps*self.batch_size/self.avg:.1f}\n')
        self.last_step = run_context.original_args().cur_step_num


class LossMonitor(Callback):
    """
    Monitor the loss in training.

    Args:
        per_print_times (int): Print the loss each every time. Default: 50.

    """

    def __init__(self, per_print_times=50):
        super(LossMonitor, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self.avg = 7.4

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        cur_step_in_epoch = (cb_params.cur_step_num -
                             1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("epoch: {} step: {}. Invalid loss, terminating training.".format(
                cb_params.cur_epoch_num, cur_step_in_epoch))

        self.avg = self.avg * 0.9996 + loss * 0.0004
        if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0 and device_id == 0:
            print("epoch: %s\t step: %s\t, avg loss is %s, loss is %s" % (
                cb_params.cur_epoch_num, cur_step_in_epoch, self.avg, loss), flush=True)

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0 and device_id == 0:
            print("epoch: %s\t loss is %s" % (
                cb_params.cur_epoch_num, loss), flush=True)

def get_lr(global_step,
           total_epochs,
           steps_per_epoch,
           lr_init=0.00001,
           lr_max=0.2,
           warmup_epochs=5):    
    """
    Generate learning rate array.
    This function should be modified according to our report.

    Args:
        global_step (int): Initial step of training.
        total_epochs (int): Total epoch of training.
        steps_per_epoch (float): Steps of one epoch.
        lr_init (float): Initial learning rate. 
        lr_max (float): Maximum learning rate.
        warmup_epochs (int): The number of warming up epochs. 

    Returns:
        np.array, learning rate array.
    """
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    lr = lr_init

    if warmup_steps != 0:
        inc_each_step = (float(lr_max) - float(lr_init)) / float(warmup_steps)
    else:
        inc_each_step = 0
    for i in range(int(total_steps)):
        if i < warmup_steps:
            lr = float(lr_init) + inc_each_step * float(i)
        else:
            if i % (2*steps_per_epoch) == 0:  # should be modified
                lr *= 0.94

        lr_each_step.append(lr)

    current_step = global_step
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[current_step:]

    return learning_rate


def Inceptionv4_train():
    """
    Train Inceptionv4 in data parallelism 

    """

    epoch_size = EPOCH_SIZE
    batch_size = 128
    class_num = 1000
    local_data_path = '/cache/data'

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(device_id=device_id)
    context.set_context(enable_graph_kernel=True)

    if device_num > 1:
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          mirror_mean=True)

    # data download
    mox.file.copy_parallel(src_url="obs://public-obs2020/pytorch-imagenet/", dst_url=local_data_path)


    # create dataset
    train_dataset = create_dataset(dataset_path=local_data_path, do_train=True,
                                   repeat_num=epoch_size, batch_size=batch_size)
    eval_dataset = create_dataset(dataset_path=local_data_path, do_train=False,
                                  repeat_num=1, batch_size=batch_size)
    train_step_size = train_dataset.get_dataset_size()

    # create model
    net = Inceptionv4(classes=class_num)

    # load checkpoint
    if USE_CKPT:
        mox.file.copy(src_url=CKPT_ROOT+META_FILE,
                      dst_url=os.path.join(local_data_path, META_FILE))
        mox.file.copy(src_url=CKPT_ROOT+CKPT_FILE,
                      dst_url=os.path.join(local_data_path, CKPT_FILE))
        load_checkpoint(os.path.join(local_data_path, CKPT_FILE), net=net)

    loss = SoftmaxCrossEntropyWithLogits(
        sparse=True, smooth_factor=0.1, num_classes=class_num, reduction="mean", is_grad=False)
    lr = Tensor(get_lr(global_step=0, total_epochs=epoch_size,
                       steps_per_epoch=train_step_size))
    opt = RMSProp(net.trainable_params(), lr, decay=0.9, epsilon=1.0)

    if device_id == 0:
        print(lr)
        print(train_step_size)
        print(eval_dataset.get_dataset_size())
    
    model = Model(net, loss_fn=loss, optimizer=opt, metrics={
                  'acc', 'top_1_accuracy', 'top_5_accuracy'})

    # define callbacks
    performance_cb = PerformanceCallback(batch_size)
    loss_cb = LossMonitor()
    config_ck = CheckpointConfig(
        save_checkpoint_steps=1500, keep_checkpoint_max=50)
    ckpoint_cb = ModelCheckpoint(prefix=CKPT_PREFIX, directory=os.path.join(
        local_data_path, "ckpt"), config=config_ck)

    if device_num == 1 or device_id == 0:
        if MEASURE_PERFMANCE:
            callbacks = [performance_cb]
        else:
            callbacks = [loss_cb, performance_cb, ckpoint_cb]
    else:
        callbacks = []

    # train model
    model.train(epoch_size, train_dataset,
                callbacks=callbacks, dataset_sink_mode=True)

    # do evaluation
    if device_num == 1 or device_id == 0:
        print(f'Start run evaluation.')
        output = model.eval(eval_dataset, dataset_sink_mode=True)
        print(f'Evaluation result: {output["acc"]}.')
        print(f'TOP1: {output["top_1_accuracy"]}.')
        print(f'TOP5: {output["top_5_accuracy"]}.')

    # save checkpoint files
    mox.file.copy_parallel(src_url=os.path.join(
        local_data_path, "ckpt"), dst_url=CKPT_ROOT)


if __name__ == '__main__':

    Inceptionv4_train()
    print(CKPT_PREFIX)
    print('Inceptionv4 training success!')
