from multiprocessing import Process, Queue, Value
import queue
import numpy as np

STEP_STARTED = 1
PROCESS = 2
STEP_ENDED = 3
DISCONNECTED = 4

STEP_SACRUM = 1
STEP_LEVEL_COUNT = 2
STEP_FACET = 3


class Data:
    timestamp = None
    image = None
    matrix = None
    spacing = None
    data_type = None

    def __init__(self, timestamp=None, image=None, matrix=None, spacing=None, sensor_data=None, status=0, step=0,
                 valid=True):

        self.valid = valid
        self.status = status
        self.step = step

        self.timestamp = timestamp
        self.image = image
        self.sensor_data = sensor_data
        self.position_matrix = matrix
        self.spacing = spacing


class GenericServer(object):
    _proc = None
    _server_address = ""
    _server_port = None

    def __init__(self):
        self._img_queue = Queue()
        self._sensor_queue = Queue()
        self._proc_flag = Value('i', 0)
        self.is_receiver_on = Value('i', 1)

    def init(self):
        self._img_queue = Queue()
        self._sensor_queue = Queue()

    def set_address(self, address, port):
        self._server_address = address
        self._server_port = port

    def start(self):
        raise NotImplementedError()

    def kill(self):
        raise NotImplementedError()

    def start_receiver(self):
        self.is_receiver_on.value = 1
        with self._proc_flag.get_lock():
            self._proc_flag.value = 1

        self._proc = Process(target=self._receive_loop, args=(self._img_queue, self._sensor_queue))
        self._proc.start()

    def stop_receiver(self):
        print("**** Stopping receiver ... ")
        with self._proc_flag.get_lock():
            self._proc_flag.value = 0

        if self._proc is None:
            return

        # cleaning queues before joining the thread otherwise it won't join
        while not self._img_queue.empty():
            self._img_queue.get()

        while not self._sensor_queue.empty():
            self._sensor_queue.get()

        self._img_queue.close()
        self._img_queue.join_thread()

        self._sensor_queue.close()
        self._sensor_queue.join_thread()

        print(self._img_queue.qsize())
        print(self._sensor_queue.qsize())

        self._proc.join()
        print("**** Receiver correctly stopped!")

    def send_data(self, data):
        raise NotImplementedError()

    def get_image_data(self):
        try:
            data = self._img_queue.get(True, 5)
        except queue.Empty:
            if self.is_receiver_on.value < 1:
                data = Data(status=DISCONNECTED)
            else:
                data = Data(valid=False)
        return data

    def get_any_data(self):
        try:
            data = self._img_queue.get(True, 5)
        except queue.Empty:
            try:
                data = self._sensor_queue.get(True, 5)
            except queue.Empty:
                return Data(valid=False)
        return data

    def cleanQueue(self):
        self._img_queue = Queue()

    def send_position(self, pos, quat):
        raise NotImplementedError()

    def _receive_data(self):
        raise NotImplementedError()

    def get_force_data(self):
        data = None
        try:
            data = self._sensor_queue.get(True, 5)
        except queue.Empty:
            pass

        return data

    # TODO: controlla meglio gestione socket
    def _receive_loop(self, img_queue, sensor_queue):
        print("Starting receiving loop")
        self.is_receiver_on.value = 1
        while self._proc_flag.value > 0:

            data = self._receive_data()

            # If the data is not valid continue without doing anything
            if not data.valid:
                continue

            if data.status == STEP_STARTED:
                img_queue.put(data)
                break

            # If the status of the message is "disconnected" close the receiving loop
            if data.status == DISCONNECTED:
                img_queue.put(data)
                sensor_queue.put(data)
                break

            # If the message status only indicates that the step is finished (it should not contain any data), then
            # the message is added to the data queue so that the processor will know this data does not contain any
            # image or sensor data but only an indication that the step is ended
            if data.status == STEP_ENDED:
                img_queue.put(data)
                sensor_queue.put(data)
                break

            # Add sensor data to the image queue if the input data contains sensor data
            if data.sensor_data is not None:
                sensor_queue.put(data)

            # Add image data to the image queue if the input data contains image data
            if data.image is not None:
                img_queue.put(data)

            # Free the queue if too many data are enqueued without being processed
            if sensor_queue.qsize() > 10:
                print("Sensor data dropped - possible data loss")
                sensor_queue.get()

            # Free the queue if too many data are enqueued without being processed
            if img_queue.qsize() > 10:
                print("Image data dropped - possible data loss")
                img_queue.get()

        self.is_receiver_on.value = 0
        print("Receiver loop is closed")