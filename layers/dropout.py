import numpy as np

class DropoutLayer:
    def __init__(self, keep_prob):
        """
        :param keep_prob - probability that given unit will not be dropped out
        """
        self._keep_prob = keep_prob
        self._mask = None

    def forward(self, a_prev: np.array, training: bool = True) -> np.array:
        """
        Forward pass through dropout layer
        :param a_prev - input tensor of any shape
        :param training - whether we're in training mode (apply dropout) or inference mode
        """
        if training and self._keep_prob < 1.0:
            self._mask = (np.random.rand(*a_prev.shape) < self._keep_prob)
            return self._apply_mask(a_prev, self._mask)
        else:
            return a_prev

    def backward(self, da_curr: np.array) -> np.array:
        """
        Backward pass through dropout layer
        """
        if self._mask is not None:
            return self._apply_mask(da_curr, self._mask)
        else:
            return da_curr

    def _apply_mask(self, array: np.array, mask: np.array) -> np.array:
        """
        Apply dropout mask and scale by keep_prob
        """
        array = array.copy()
        array *= mask
        array /= self._keep_prob  # Scale to maintain expected value
        return array