from multiprocessing import Process, Queue, Value
import queue
import logging

__all__ = ['Data', 'GenericServer', 'STEP_STARTED', 'PROCESS', 'STEP_ENDED', 'DISCONNECTED', 'STEP_SACRUM',
           'STEP_LEVEL_COUNT', 'STEP_FACET']

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
    """
    Generic Server to communicate with the main C++ application controlling the robot and streaming b-mode ultrasound
    and force data
    """
    _process_thread = None
    _server_address = ""
    _server_port = None

    def __init__(self):
        self._img_queue = Queue()
        self._sensor_queue = Queue()
        self._process_flag = Value('i', 0)

    def set_address(self, address, port):
        """
        Sets the server address and port
        :param str address: The server address
        :param int port: The server port
        """
        self._server_address = address
        self._server_port = port

    def start(self):
        """
        Interface to the function that start the server on the desired address and port
        """
        raise NotImplementedError

    def stop(self):
        """
        Interface to the function that stops the server
        """
        raise NotImplementedError()

    def listen_for_connection(self):
        """
        Interface to the function that listens for incoming client connections
        """
        raise NotImplementedError()

    def kill_clients_connection(self):
        """
        Interface to the function that kills existing client connections
        """
        raise NotImplementedError

    def start_receiver(self):
        """
        Starts the receiver loop on a separate process. If the receiver loops is already running it does nothing and
        returns False. Otherwise it joins in a clean way pre-existing process if it exists, cleans the data queues and
        start the process.
        :return: True if the process was correctly launched, false otherwise
        """

        # If the receiver process is running return False
        if self._process_flag.value > 0:
            logging.info("process thread already running - If you want to restarted you need to stop it beforehand")
            return False

        # If the receiver process is not running (it is out of the receiving loop) then clean close the thread and empty
        # the queues
        if self._process_thread is not None:
            self.stop_receiver()

        # Instantiate and run the receiver process
        with self._process_flag.get_lock():
            self._process_flag.value = 1

        self._process_thread = Process(target=self._receive_loop, args=(self._img_queue, self._sensor_queue))
        self._process_thread.start()
        logging.info("Receiver Process Correctly started")
        return True

    def stop_receiver(self):
        """
        Stops the receiving process and cleans the processing queues
        """
        logging.info("Stopping Receiver Process ... ")
        with self._process_flag.get_lock():
            self._process_flag.value = 0

        if self._process_thread is None:
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

        self._process_thread.join()
        self._process_thread = None
        logging.info("... Receiver Process correctly stopped!")

    def get_image_data(self):
        """
        Returns the first Data object stored in the image data queue. If the queue is empty it waits for 5 seconds for
        new data to be added to the queue. If no data are added and the receiver process is active, it returns a Data
        object with "valid" field equal to False. If no data are added and the receiver process is not active, it
        returns a Data object with DISCONNECTED STATUS.
        :return: The Data object stored in the queue if the queue is not empty. A Data object with "status" DISCONNECTED
        if the receiver process is not running and the queue is empty, a Data object with "valid" False if the receiver
        process is running and the queue is empty.
        """
        try:
            data = self._img_queue.get(True, 5)
        except queue.Empty:
            if self._process_flag.value < 1:
                data = Data(status=DISCONNECTED)
            else:
                data = Data(valid=False)
        return data

    def get_force_data(self):
        """
        Returns the first Data object stored in the force data queue. If the queue is empty it waits for 5 seconds for
        new data to be added to the queue. If no data are added and the receiver process is active, it returns a Data
        object with "valid" field equal to False. If no data are added and the receiver process is not active, it
        returns a Data object with DISCONNECTED STATUS.
        :return: The Data object stored in the queue if the queue is not empty. A Data object with "status" DISCONNECTED
        if the receiver process is not running and the queue is empty, a Data object with "valid" False if the receiver
        process is running and the queue is empty.
        """
        data = None
        try:
            data = self._sensor_queue.get(True, 5)
        except queue.Empty:
            if self._process_flag.value < 1:
                data = Data(status=DISCONNECTED)
            else:
                data = Data(valid=False)
        return data

    def get_all_data(self):
        """
        Gets both image and force Data stored in the image and force data queue
        :return: Available data in the image and force data queue
        """
        try:
            img_data = self._img_queue.get(True, 5)
        except queue.Empty:
            if self._process_flag.value < 1:
                img_data = Data(status=DISCONNECTED)
            else:
                img_data = Data(valid=False)

        try:
            force_data = self._img_queue.get(True, 5)
        except queue.Empty:
            if self._process_flag.value < 1:
                force_data = Data(status=DISCONNECTED)
            else:
                force_data = Data(valid=False)

        return img_data, force_data

    def send_data(self, data):
        raise NotImplementedError()

    def send_position(self, pos, quat):
        raise NotImplementedError()

    def receive_data(self):
        raise NotImplementedError()

    def _receive_loop(self, img_queue, sensor_queue):
        logging.info("Launching Receiver Process")
        while self._process_flag.value > 0:

            data = self.receive_data()

            # If the data is not valid continue without doing anything
            if not data.valid:
                continue

            if data.status == STEP_STARTED:
                img_queue.put(data)
                sensor_queue.put(data)
                continue

            # If the message status only indicates that the step is finished (it should not contain any data), then
            # the message is added to the data queue so that the processor will know this data does not contain any
            # image or sensor data but only an indication that the step is ended
            if data.status == STEP_ENDED:
                img_queue.put(data)
                sensor_queue.put(data)

            # If the status of the message is "disconnected" close the receiving loop
            if data.status == DISCONNECTED:
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

        with self._process_flag.get_lock():
            self._process_flag.value = 0
