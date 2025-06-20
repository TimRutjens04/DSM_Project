import numpy as np

class FlattenLayer:
    def __init__(self):
        self._shape = None

    def forward(self, a_prev: np.array) -> np.array:
        """
        Flatten input tensor
        :param a_prev - input tensor (H, W, C) for single image or (N, H, W, C) for batch
        :return - flattened tensor
        """
        self._shape = a_prev.shape
        
        if len(a_prev.shape) == 3:  # Single image (H, W, C)
            return a_prev.flatten()
        elif len(a_prev.shape) == 4:  # Batch (N, H, W, C)
            return a_prev.reshape(a_prev.shape[0], -1)
        else:
            return a_prev  # Already flattened

    def backward(self, da_curr: np.array) -> np.array:
        """
        Reshape gradient back to original shape
        :param da_curr - flattened gradient tensor
        :return - gradient tensor with original shape
        """
        return da_curr.reshape(self._shape)