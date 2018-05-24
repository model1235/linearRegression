#展示了content-length的作用

import time
import socketserver
class MyServer(socketserver.BaseRequestHandler):
    def handle(self):
        conn = self.request
        # conn.sendall(bytes('我是多线程',encoding="utf-8"))
        data = conn.recv(1024)
        print("收到%s" % (data))
        head = "HTTP/1.1 200 OK\nServer: nginx/1.13.7\nContent-Type: text/html\nContent-Length: 15\nConnection: keep-alive\n\n"
        conn.sendall(bytes(head, encoding="utf-8"))
        Flag = True
        while Flag:
            conn.sendall(bytes("a", encoding="utf-8"))
            time.sleep(3)
            # if data == 'exit':
            #     Flag = False
            # elif data == '0':
            #     conn.sendall(bytes('您输入的是0',encoding="utf-8"))
            # else:
            #     conn.sendall(bytes('请重新输入.',encoding="utf-8"))
if __name__ == '__main__':
    server = socketserver.ThreadingTCPServer(('127.0.0.1',8009),MyServer)
    server.serve_forever()

