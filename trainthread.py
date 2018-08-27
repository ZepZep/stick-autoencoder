import multiprocessing


class TrainThread(multiprocessing.Process):
    def __init__(self, in_queue, out_queue):
        super(TrainThread, self).__init__()
        self.in_queue: multiprocessing.Queue = in_queue
        self.out_queue: multiprocessing.Queue = out_queue

    def run(self):
        self._init_data()
        msg = self.in_queue.get()
        rep, ep = (int(x) for x in msg.split("/"))


    def _init_data(self):
        pass

    def _train(self, rep, ep):
        pass


