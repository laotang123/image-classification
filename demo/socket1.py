# -*- coding: utf-8 -*-
# @Time    : 2019/4/28 13:52
# @Author  : ljf
import socket
import os

HOST = "192.168.1.10"  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    # 绑定监听
    s.bind((HOST, PORT))
    s.listen() # 监听
    conn, addr = s.accept() # 接收地址和内容
    print(conn)
    with conn:
        print("Connected by", addr)
        # 持续监听状态，
        while True:
            data = conn.recv(1024)
            print(data)
            print(type(data))
            data = data.decode("utf-8")

            decode_data = data.split()
            with open("system.sh", mode="r", encoding="utf-8") as file:
                bash = file.read()
            for item in decode_data:
                bash+=" "+item
            print(bash)
            os.system(bash)
            if not data:
                break
            # 传输epoch实时数据
            conn.sendall(b"jdfaksj")


# print(bash)
# os.system(bash)

