from multiprocessing import Queue, Process, Value
import numpy as np
import os
import pandas as pd
import imfusion
import time

imfusion_be = 0
pandas_be = 1


class ForceUSSimulator(object):

    def __init__(self):
        self._us_process = None
        self._force_process = None
        self._pose_process = None

        self._img_queue = Queue()
        self._force_queue = Queue()
        self._pose_queue = Queue()
        self._proc_flag = Value('i', False)

        self.force_reader = ImFusionForce()
        self.ultrasound_reader = ImfusionUsSweep()
        self.pose_reader = ImFusionPose()

    def load_data(self, data_path):
        sweep = imfusion.open(data_path)

        self.ultrasound_reader.load_data(sweep)

        if "Force" in sweep.tracking(0).name:
            self.force_reader.load_data(sweep.tracking(0))
            self.pose_reader.load_data(sweep.tracking(1))
        else:
            self.force_reader.load_data(sweep.tracking(1))
            self.pose_reader.load_data(sweep.tracking(0))

    def get_img_data(self):
        try:
            ts, img = self._img_queue.get(timeout=1)
        except self._img_queue.empty():
            ts, img = -1, None

        return ts, img

    def get_force_data(self):
        try:
            ts, force = self._force_queue.get(timeout=1)
        except self._force_queue.empty():
            ts, force = -1, None

        return ts, force

    def get_pose_data(self):
        try:
            ts, pose = self._pose_queue.get(timeout=1)
        except self._pose_queue.empty():
            ts, pose = -1, None

        return ts, pose

    def launch(self):

        self._us_process = Process(target=self.data_reader, args=(self.ultrasound_reader, self._img_queue, self._proc_flag))
        self._force_process = Process(target=self.data_reader, args=(self.force_reader, self._force_queue, self._proc_flag))
        self._pose_process = Process(target=self.data_reader, args=(self.pose_reader, self._pose_queue, self._proc_flag))

        self._proc_flag = True
        self._us_process.start()
        self._force_process.start()
        self._pose_process.start()

        return True

    def join_processes(self):
        self._proc_flag = False
        self.clean_queues()
        self._us_process.join()
        self._force_process.join()
        self._pose_process.join()

    def clean_queues(self):
        self.clear_queue(self._force_queue)
        self.clear_queue(self._pose_queue)
        self.clear_queue(self._img_queue)

    @staticmethod
    def clear_queue(q):
        while not q.empty():
            q.get_nowait()

    @staticmethod
    def data_reader(reader, data_queue, flag):
        dt = 1/reader.get_avg_freq()

        while flag.value:
            ret, timestamp, data = reader.get_next_data()

            if not ret:
                break

            data_queue.put((timestamp, data))

            # only the most recent value is left in the queue
            if data_queue.qsize() > 1:
                data_queue.get()
            time.sleep(dt)

        # Indicate that no more data will be put on this queue by the current process
        data_queue.close()

        #  It blocks until the background thread exits, ensuring that all data in the buffer has been flushed to the
        #  pipe. It blocks until the queue is empty (all data have been consumed) and joins
        data_queue.join_thread()


class GenericReader(object):
    """
    :ivar ndarray _data: Data - they must be a N x * array where N is the sequence length.
    """
    def __init__(self):
        self._timestamp = np.array(0)
        self._data = np.array(0)
        self._data_idx = 0
        self._data_len = 0
        self._avg_freq = 0

    def load_data(self, data_path):
        self._load_data(data_path)
        self._compute_freq()

    def seq_len(self):
        return self._data_len

    def get_nex_data(self):

        if self._data_idx > self._data_len:
            return False, None, None

        next_timestamp = self._timestamp[self._data_idx]
        next_data = np.squeeze(self._data[self._data_idx])
        self._data_idx += 1

        return True, next_timestamp, next_data

    def get_avg_freq(self):
        return self._avg_freq

    def _compute_freq(self):
        dt_array = np.diff(self._timestamp)
        freq_array = 1/dt_array
        self._avg_freq = np.mean(freq_array)

    def _load_data(self, data_path):
        raise NotImplementedError


class ImFusionForce(GenericReader):
    def __init__(self):
        super(GenericReader, self).__init__()

    def _load_data(self, tracking_instrument):

        self._data_len = tracking_instrument.size()
        self._data = np.zeros([self._data_len, 3])
        self._timestamp = np.zeros(self._data_len)
        for i in range(self._data_len):
            self._data[i] = tracking_instrument.matrix(i)[0:3, 3]
            self._timestamp[i] = tracking_instrument.timestamp(i)


class ImFusionPose(GenericReader):
    def __init__(self):
        super(GenericReader, self).__init__()

    def _load_data(self, tracking_instrument):

        self._data_len = tracking_instrument.size()
        self._data = np.zeros([self._data_len, 3])
        self._timestamp = np.zeros(self._data_len)
        for i in range(self._data_len):
            self._data[i] = tracking_instrument.matrix(i)
            self._timestamp[i] = tracking_instrument.timestamp(i)


class ImfusionUsSweep(GenericReader):
    def __init__(self):
        super(GenericReader, self).__init__()

    def _load_data(self, sweep):

        self._data = np.squeeze(np.array(sweep))
        self._data_len = sweep.shape[0]
        self._timestamp = np.zeros(self._data_len)
        for i in range(self._data_len):
            self._timestamp[i] = sweep.timestamp(i)


class CsvForceReader(GenericReader):
    def __init__(self):
        super(GenericReader, self).__init__()

    def _load_data(self, data_path):
        if not os.path.exists(data_path):
            raise RuntimeError("Data path not found in {}: not existing dir: {}".format(self.__name__, data_path))

        dframe = pd.read_csv(data_path)

        # if there is only one column probably the delimiter is ;
        if len(dframe.columns) < 2:
            dframe = pd.read_csv(data_path, delimiter=";")

        if "y_force" in dframe.columns:
            self._timestamp = np.array(dframe["timestamp"])
            self._data = np.array(dframe["y_force"])
        else:
            self._timestamp = np.zeros(0)
            self._data = np.zeros(0)

        if self._timestamp.shape[0] != self._data.shape[0]:
            raise RuntimeError("Force and Timestamp array must have equal size")

        self._data_len = self._data.shape[0]
