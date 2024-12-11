# This class will be used to keep track of evaluations of cost functions
class EvaluationTracker():
    def __init__(self, max_eval = 3000):
        self.currentEvalCount = 0
        self.maxEval = max_eval
