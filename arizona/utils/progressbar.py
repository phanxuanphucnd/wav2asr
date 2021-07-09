import progressbar

class MyProgressBar():
    """A class custom ProgressBar"""
    def __init__(self, name):
        self.pbar = None
        self.name = name

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            # widgets = ['Downloading: ', progressbar.Bar(marker=progressbar.RotatingMarker()), ' ', progressbar.Percentage()," | ", progressbar.ETA(), " | "]
            widgets = [f'\x1b[33mDownloading: {self.name} \x1b[39m', progressbar.Percentage(), ' ', progressbar.Bar(marker='█')," | ", progressbar.ETA(), " | "]

            self.pbar=progressbar.ProgressBar(widgets=widgets ,maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()
