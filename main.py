import socket

# command: .\TrackmaniaServer.exe /game_settings=MatchSettings/Nations/NationsBlue.txt /dedicated_cfg=dedicated_cfg.txt # noqa: E501


class GameBridge:
    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(10.0)
        self.socket.connect(('localhost', 5000))
        self.reqhandle = 0x80000000

        self.cb_checkpoint = None

    def initialize(self):
        print("Handshake...")
        size = int.from_bytes(self.socket.recv(4), "little")
        protocol = self.socket.recv(size).decode("ascii")
        if protocol != "GBXRemote 2":
            raise RuntimeError(f"Invalid protocol: {protocol}")
        print("Handshake done!")

    def send(self, xml: str):
        length = len(xml).to_bytes(4, "little")
        self.reqhandle += 1
        handle = self.reqhandle.to_bytes(4, "little")
        payload = xml.encode("ascii")
        data = length + handle + payload
        # print(''.join('{:02x}\n'.format(x) for x in data))
        self.socket.sendall(data)

    def recv(self):
        size = int.from_bytes(self.socket.recv(4), "little")
        handle = int.from_bytes(self.socket.recv(4), "little")
        data = self.socket.recv(size).decode("ascii")
        print(f"Received message {handle & 0x80000000}")
        return data

    def enable_callbacks(self):
        pass

    def close(self):
        self.socket.close()


msg1 = '<?xml version="1.0" encoding="utf-8" ?><methodCall><methodName>system.listMethods</methodName><params></params></methodCall>'  # noqa: E501
enable_callbacks = '<?xml version="1.0" encoding="utf-8" ?><methodCall><methodName>EnableCallbacks</methodName><params><param><value><boolean>1</boolean></value></param></params></methodCall>'  # noqa: E501

if __name__ == "__main__":
    gb = GameBridge()
    gb.initialize()
    # gb.send(msg1)
    gb.send(enable_callbacks)
    while True:
        try:
            print(gb.recv())
        except socket.timeout:
            pass
        except KeyboardInterrupt:
            break
    gb.close()
