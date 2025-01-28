class Pipeline:
    def __init__(self, model_name):
        self.model_name = model_name

    def get_answer(self, question):
        raise NotImplementedError
    