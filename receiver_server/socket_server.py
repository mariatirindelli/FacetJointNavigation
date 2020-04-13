import socket
from receiver_server import GenericServer
import logging

__all__ = ['SocketServer']

#  very simple server with 2 socket open: one for data stream and the other for commands


class SocketServer(GenericServer):
    _server_socket = None
    _data_socket = None
    _cmd_socket = None

    def __init__(self):
        GenericServer.__init__(self)

    def start(self):

        # Bind the server socket to the server address
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.bind((self._server_address, self._server_port))
        logging.info("Server started at: {} : {}".format(self._server_address, self._server_port))

    def stop(self):
        logging.info("shutting down connection")

        if self._data_socket is not None:
            self._data_socket.shutdown(socket.SHUT_RDWR)
            self._data_socket.close()

        if self._cmd_socket is not None:
            self._cmd_socket.shutdown(socket.SHUT_RDWR)
            self._cmd_socket.close()

        if self._server_socket is not None:
            self._server_socket.shutdown(socket.SHUT_RDWR)
            self._server_socket.close()

        logging.info("Server shut down")

    def listen_for_connection(self):
        logging.info("Listening for connection")

        # listen for 2 incoming connections
        self._server_socket.listen(2)

        # Accept the incoming connections
        in_socket_1, address_1 = self._server_socket.accept()
        self._handshake(in_socket_1, address_1)

        in_socket_2, address_2 = self._server_socket.accept()
        self._handshake(in_socket_2, address_2)

    def kill_clients_connection(self):
        logging.info("shutting down client connection")

        try:
            self._data_socket.shutdown(socket.SHUT_RDWR)
            self._data_socket.close()
        except:
            print("data socket already disconnected")

        try:
            self._cmd_socket.shutdown(socket.SHUT_RDWR)
            self._cmd_socket.close()
        except:
            print("command socket already disconnected")
        print("Connection are shut down.. listening for new connections")

    def setTimeout(self, timeout=0):
        if self._data_socket is not None:
            self._data_socket.settimeout(timeout)
            self._cmd_socket.settimeout(timeout)

    def send_data(self, data):
        if not isinstance(data, (bytes, bytearray)):
            return
        self._cmd_socket.sendall(data)

    def send_position(self, pos, quat):
        data = bytes(pos) + bytes(quat)
        self.send_data(data)

    def receive_data(self):
        pass

    def _handshake(self, s, address=None):
        return

    @staticmethod
    def _recvall(s, n):
        # Helper function to recv n bytes or return None if EOF is hit
        raw_data = B''
        while len(raw_data) < n:
            packet = s.recv(n - len(raw_data))
            if not packet:
                return False, raw_data
            raw_data += packet
        return True, raw_data
