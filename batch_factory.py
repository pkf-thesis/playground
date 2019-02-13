class BatchFactory:

    def __init__(self, batch_size):

        self.music_ids = {}

        # Keep track of completed cycles through data
        self.epoc_count = 0
    
    def add_data(self):
        raise NotImplementedError

    def next(self):
        raise NotImplementedError
    

    def shuffle_data(self):
        raise NotImplementedError
