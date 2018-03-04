import sys
import socket
import logging
from concurrent.futures import ThreadPoolExecutor
logging.basicConfig(
    format='[%(asctime)s] [%(levelname)s] [%(processName)s] [%(threadName)s] : %(message)s',
    level=logging.INFO)


def argv():
    server_host, server_port, url_img = ('localhost', 50001, "https://www.collinsdictionary.com/images/thumb/giraffe_114856285_250.jpg")
    if len(sys.argv) == 4: server_host, server_port, url_img = (sys.argv[1], int(sys.argv[2]), sys.argv[3])
    return server_host, server_port, url_img


def worker():
    server_host, server_port, url_img = argv()
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.connect((server_host, server_port))
    logging.info("Connected to server at (%s, %d)" % (server_host, server_port))
    msg = url_img + "[END]"
    soc.sendall(msg.encode('utf-8'))
    logging.info("URL sent to the server")
    data = ""
    while 1:
        buf = soc.recv(2048)
        if not buf:
            soc.close()
            break
        else:
            data += buf.decode('utf-8')

    logging.info("Server response: %s" % data)


def run():
    #with ThreadPoolExecutor(max_workers=5) as executor:
    #    [executor.submit(worker) for _ in range(300)]
    worker()

if __name__ == '__main__':
    run()

