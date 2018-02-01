import socket
import sys
import os

PORT = 8888
CHUNK = 4096

def main(args):
    if len(args) < 3:
        print("Usage: " + args[0] + "<server_ip_addr> <image1> [<image2> ...]")
        return
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args[1],PORT))
    data_images = ""
    for i in range(2,len(args)):
        tokens = args[i].split(".")
        if len(tokens) > 0:
            img_type = tokens[len(tokens)-1]
        else:
            img_type = "na"
        img_size = os.path.getsize(args[i])
        data_images = data_images + img_type + "," + str(img_size) + ";"
    try:
        sock.send(str.encode(data_images))
        data = sock.recv(1024).decode()
        n = 2
        while data == "OK":
            if n >= len(args):
                break
            with open(args[n], 'rb') as f:
                data = f.read()
                print("Image size [bytes] in memory: " + str(len(data)))
                for i in range(len(data) // CHUNK):
                    start = i * CHUNK
                    end = start + CHUNK
                    nbytes = sock.send(data[start:end])
#                    print("Bytes sent on socket: " + str(nbytes))
                start = start + CHUNK
                end = len(data)
                if (end >= start):
                    nbytes = sock.send(data[start:end])
#                    print("Bytes sent on socket: " + str(nbytes))                
                data = sock.recv(1024).decode()
                f.close()
            n = n + 1
    finally:
       sock.close()
    print("Done.")

if __name__ == "__main__":
    main(sys.argv)
