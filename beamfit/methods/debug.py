from ..base import AnalysisMethod


class AnalysisMethodDebugger(AnalysisMethod):
    """
    Analysis method with `fit` that just passes the image out of it to test image pre-processing functions.
    """

    def __init__(self, **kwargs):
        super(AnalysisMethodDebugger, self).__init__(**kwargs)

    def __fit__(self, image, image_sigmas=None):
        return image
