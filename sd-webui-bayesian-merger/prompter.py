from pathlib import Path


class Prompter():
    def __init__(self, wildcards_dir:str):
        self.find_wildcards(wildcards_dir)

    def find_wildcards(self, wildcards_dir: str) -> None:
        wdir = Path(wildcards_dir)
        if self.wildcards_dir.exists():
            self.wildcards = {w.stem: w for w in self.wdir.glob('*.txt')}
        else:
            self.widcards = {}

        