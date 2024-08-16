import time

class Chronometer:

    def __init__(self):
        self.stored_times = []

    def start(self):
        self.t = time.time()

    def stop(self, message):
        self.stored_times.append([message, time.time() - self.t])

    def __str__(self):
        s = ""
        for times in self.stored_times:
            s += times[0] + ": " + str(round(times[1], 3)) + "s\n"
        return s