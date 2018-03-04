#"pillow", "requests", "NumPy downgrade to ver 1.13 "
import tensorflow as tf
import numpy as np
from keras_squeezenet import SqueezeNet
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import logging
import socket
import requests
import os
import sys
import uuid

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(
    format='[%(asctime)s] [%(levelname)s] [%(processName)s] [%(threadName)s] : %(message)s',
    level=logging.INFO)


def remove_file(img_path):
    if os.path.isfile(img_path): os.remove(img_path)


def download_img(url):
    img_dir = "images"
    img_path = "images/%s.jpg" % str(uuid.uuid4())

    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    elif os.path.exists(img_dir):
        with open(img_path, 'wb') as handler:
            handler.write(requests.get(url).content)
            logging.info("Image saved to %s" % img_path)

    return img_path


def thread_worker(client_socket, client_address, graph, model):
    logging.info("Received Client ('%s', %d)." % (client_address[0], client_address[1]))
    buf_data = ""
    while 1:
        buf = client_socket.recv(1024)
        if not buf:
            break
        else:
            buf_data += buf.decode('utf-8')

        if "[END]" in buf_data:
            buf_data = buf_data[:buf_data.find("[END]")]
            logging.info("Client submit URL %s" % buf_data)

            with graph.as_default():
                img_path = download_img(buf_data)
                img = image.load_img(img_path, target_size=(227, 227))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                prediction = model.predict(x)
                result = decode_predictions(prediction)
                object = result[0][0][1]
                probability = "%.3f" % result[0][0][2]
                send_result = "(\"%s\", %s)" % (object, probability)

            logging.info("SqueezeNet result: %s" % send_result)
            client_socket.sendall(send_result.encode('utf-8'))
            client_socket.shutdown(socket.SHUT_RDWR)
            logging.info("Client disconnected.")
            client_socket.close()
            remove_file(img_path)
            break


def child_process(queue, lock):
    graph = tf.get_default_graph()
    model = SqueezeNet()
    with ThreadPoolExecutor(max_workers=4) as executor:
        while 1:
            if not queue.empty():
                with lock:
                    client_socket, client_address = queue.get()
                executor.submit(thread_worker, client_socket, client_address, graph, model)


def argv():
    server_port, num_processes = (50001, 4)
    if len(sys.argv) == 3: server_port, num_processes = (int(sys.argv[1]), int(sys.argv[2]))
    return server_port, num_processes


def run_server():
    server_port, num_processes = argv()
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', server_port))
    server_socket.listen(10)
    logging.info("Start listening for connections on port %d" % server_port)

    socket_queue = multiprocessing.Queue()
    lock = multiprocessing.Lock()
    processes = []

    for _ in range(num_processes):
        p = multiprocessing.Process(target=child_process, args=(socket_queue,lock, ))
        logging.info("Created process %s" % p.name)
        processes.append(p)

    [p.start() for p in processes]

    while 1:
        new_socket = server_socket.accept()
        logging.info("Client ('%s', %d) connected." % (new_socket[1][0], new_socket[1][1]))
        socket_queue.put(new_socket)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    run_server()