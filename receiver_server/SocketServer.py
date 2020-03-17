import socket
from receiver_server import GenericServer, Data
import logging

#  very simple server with 2 socket open: one for data stream and the other for commands


class SocketSpineServer(GenericServer):
    _server_socket = None
    _data_socket = None
    _cmd_socket = None

    def __init__(self):
        GenericServer.__init__(self)

    def start(self):

        # Bind the server socket to the server address
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.bind((self._server_address, self._server_port))

    def listen_for_connection(self):
        print("listening for connections")

        # listen for 2 incoming connections
        self._server_socket.listen(2)

        # Accept the incoming connections
        in_socket_1, address_1 = self._server_socket.accept()
        self._handshake(in_socket_1, address_1)

        in_socket_2, address_2 = self._server_socket.accept()
        self._handshake(in_socket_2, address_2)


    def setTimeout(self, timeout=0):
        if self._data_socket is not None:
            self._data_socket.settimeout(timeout)
            self._cmd_socket.settimeout(timeout)

    def kill_client_sockets(self):
        logging.info("shutting down connection")

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

    def kill(self):
        logging.info("shutting down connection")

        self._data_socket.shutdown(socket.SHUT_RDWR)
        self._data_socket.close()

        self._cmd_socket.shutdown(socket.SHUT_RDWR)
        self._cmd_socket.close()

        self._server_socket.shutdown(socket.SHUT_RDWR)
        self._server_socket.close()

    def _handshake(self, s, address=None):
        return

    def _receive_data(self):
        pass

    def send_data(self, data):
        self._cmd_socket.sendall(data)

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
