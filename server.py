#!/usr/bin/env python3

import socket

HOST = ''  # Standard loopback interface address (localhost)
PORT = 3030        # Port to listen on (non-privileged ports are > 1023)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen()
conn, addr = s.accept()
print('Connected by', addr)
while True:
    data = conn.recv(3)
    if repr(data) == "b'exp'": # sempre vem com o b'str'
        print("Expressao")
    if not data:
        print("Sem dados. Cliente encerrou a conexao.")
        break
    print("data:", data)
    print("repr(data):", repr(data))
    conn.sendall(data)