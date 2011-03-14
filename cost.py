"""
Cost classes: classes that encapsulate the cost evaluation for the DAE
training criterion.
"""
# Standard library imports
from itertools import imap

# Third-party imports
from theano import tensor

class SupervisedCost(object):
    """
    A cost object is allocated in the same fashion as other
    objects in this file, with a 'conf' dictionary (or object
    supporting __getitem__) containing relevant hyperparameters.
    """
    def __init__(self, conf, model):
        self.conf = conf
        # TODO: Do stuff depending on conf parameters (for example
        # use different cross-entropy if act_end == "tanh" or not)
        self.model = model

    def __call__(self, *inputs):
        """Symbolic expression denoting the reconstruction error."""
        raise NotImplementedError()

class MeanSquaredError(SupervisedCost):
    """
    Symbolic expression for mean-squared error between the target
    and a prediction.
    """
    def __call__(self, prediction, target):
        msq = lambda p, t: ((p - t)**2).sum(axis=1).mean()
        if isinstance(prediction, tensor.Variable):
            return msq(prediction, target)
        else:
            # TODO: Think of something more sensible to do than sum(). On one
            # hand, if we're treating everything in parallel it should return
            # a list. On the other, we need a scalar for everything else to
            # work.

            # This will likely get refactored out into a "costs" module or
            # something like that.
            return sum(imap(msq, prediction, target))

class CrossEntropy(SupervisedCost):
    """
    Symbolic expression for elementwise cross-entropy between target
    and prediction. Use for binary-valued features (but not for,
    e.g., one-hot codes).
    """
    def __call__(self, prediction, target):
        ce = lambda x, z: x * tensor.log(z) + (1 - x) * tensor.log(1 - z)
        if isinstance(prediction, tensor.Variable):
            return ce(prediction, target)
        return sum(
            imap(lambda p, t: ce(p, t).sum(axis=1).mean(), prediction, target)
        )

##################################################
def get(str):
    """ Evaluate str into a cost object, if it exists """
    obj = globals()[str]
    if issubclass(obj, SupervisedCost):
        return obj
    else:
        raise NameError(str)
