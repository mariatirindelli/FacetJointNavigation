from receiver_server import *
from step_processors import *
import logging


ckpt_path = "/media/maria/Elements/Maria/Submissions/IROS2020/Experiments/models/ResNet18.pt"
mstcn_ckpt_path = "/media/maria/Elements/Maria/Submissions/IROS2020/Experiments/models/TCN_imgProb.ckpt"
DESIRED_LEVEL = 4


def main(param):

    receiving_server = IGTLinkServer()

    receiving_server.set_address(address="127.0.0.1",
                                 port=9004)

    receiving_server.start()
    receiving_server.listen_for_connection()
    receiving_server.start_receiver()

    while True:

        data, _ = receiving_server.get_all_data()

        if not data.valid:
            continue

        if data.status == DISCONNECTED:
            receiving_server.stop_receiver()
            receiving_server.kill_clients_connection()
            receiving_server.listen_for_connection()
            continue

        if data.status == STEP_STARTED:
            step = data.step
            logging.info("Starting Step: {}".format(step))

            if step == STEP_LEVEL_COUNT:
                step = VertebraeCounting(param.cnn_ckpt, param.tcn_ckpt)
            else:
                logging.info("Unknown step")
                continue

            step.gpu_acceleration()
            step.process(receiving_server)
            logging.info("{} step ended".format(step))


main(dict())
