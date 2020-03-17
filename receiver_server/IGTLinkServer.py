from receiver_server import *
import pygtlink


class IGTLinkServer(SocketSpineServer):

    def __init__(self):
        SocketSpineServer.__init__(self)
        status_msg = pygtlink.StatusMessage()

        print("smt")

    def send_position(self, pos, quat):
        pos_msg = pygtlink.PositionMessage()
        pos_msg.setPosition(pos)
        pos_msg.setQuaternion(quat)

        pos_msg.pack()
        self.send_data(pos_msg.header + pos_msg.body)

    def _handshake(self, s, address=None):

        status_msg = pygtlink.StatusMessage()
        res, status_msg.header = self._recvall(s, pygtlink.IGTL_HEADER_SIZE)

        if not status_msg.unpack() == pygtlink.UNPACK_HEADER:
            print("Error in unpacking header")
            return None

        res, status_msg.body = self._recvall(s, status_msg.getPackBodySize())

        # if not status_msg.unpack() == pygtlink.UNPACK_BODY:
        #     print("Error in unpacking body")
        #     return None

        # if status_msg.getCode() == 0:
        #     print("Error in message: status code is 0")
        #     return

        if status_msg.getDeviceName() == "StreamerSocket":
            self._data_socket = s
            print("Connection with streamer established at ip:{}".format(address))

        elif status_msg.getDeviceName() == "CommandSocket":
            self._cmd_socket = s
            print("Connection with commander established at ip:{}".format(address))

        else:
            print("Unrecognized device name - returning with error")
            return

    def _receive_data(self):

        message = pygtlink.MessageBase()

        res, message.header = self._recvall(self._data_socket, pygtlink.IGTL_HEADER_SIZE)

        if res is False:
            return Data(status=DISCONNECTED)

        if not message.unpack() == pygtlink.UNPACK_HEADER:
            print("Error in unpacking header")
            return Data(valid=False)

        if message.getMessageType() == "IMAGE":
            img_msg = pygtlink.ImageMessage2()
            img_msg.copyHeader(message)
            res, img_msg.body = self._recvall(self._data_socket, img_msg.getPackBodySize())

            if not img_msg.unpack() == pygtlink.UNPACK_BODY:
                print("Error in unpacking image body")
                return Data(valid=False)
            img = img_msg.getData()
            matrix = img_msg.getMatrix()
            spacing = img_msg.getSpacing()
            data = Data(image=img, matrix=matrix, spacing=spacing)

        elif message.getMessageType() == "SENSOR":

            sensorMsg = pygtlink.SensorMessage()
            sensorMsg.copyHeader(message)
            res, sensorMsg.body = self._recvall(self._data_socket, sensorMsg.getPackBodySize())

            if not sensorMsg.unpack() == pygtlink.UNPACK_BODY:
                print("Error in unpacking sensor body")
                return Data(valid=False)
            data = Data(sensor_data=sensorMsg.getData())
        else:
            # TODO: handle this, would probably lead to an error
            return Data(valid=False)

        return data
