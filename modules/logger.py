import sys

class Logger():
    def __init__(self, logfile='default.log', stream=sys.stdout, mode='a'):
        self.terminal = stream
        self.log = open(logfile, mode)   # add
        if mode == 'a':
            self.log.write('\n')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass