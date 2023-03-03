from PIL import Image


class Scorer:
    def score(self, img: Image) -> float:
        # TODO: do something
        return 0.0

    def batch_score(self, images: [Image]) -> [float]:
        return [self.score(img) for img in images]

    def average_score(self, scores: [float]) -> float:
        # TODO: do something else?
        return sum(scores) / len(scores)
