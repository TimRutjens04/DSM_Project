from typing import Tuple
import numpy as np

class MaxPoolLayer:
    def __init__(self, pool_size: Tuple[int, int] = (2, 2), stride: int = 2):
        """
        Max pooling layer optimized for 64x64x3 images
        :param pool_size - tuple holding shape of 2D pooling window
        :param stride - stride along width and height
        """
        self._pool_size = pool_size
        self._stride = stride
        self._a = None
        self._cache = {}

    def forward(self, a_prev: np.array) -> np.array:
        """
        Forward pass through max pooling
        :param a_prev - input tensor (H, W, C) or (N, H, W, C)
        :return - pooled tensor
        """
        # Handle single image case
        single_image = False
        if len(a_prev.shape) == 3:
            single_image = True
            a_prev = a_prev[np.newaxis, ...]  # Add batch dimension
            
        self._a = np.array(a_prev, copy=True)
        n, h_in, w_in, c = a_prev.shape
        h_pool, w_pool = self._pool_size
        
        # Calculate output dimensions
        h_out = 1 + (h_in - h_pool) // self._stride
        w_out = 1 + (w_in - w_pool) // self._stride
        
        output = np.zeros((n, h_out, w_out, c))

        # Clear cache for new forward pass
        self._cache = {}

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self._stride
                h_end = h_start + h_pool
                w_start = j * self._stride
                w_end = w_start + w_pool
                
                # Extract slice for pooling
                a_prev_slice = a_prev[:, h_start:h_end, w_start:w_end, :]
                
                # Save mask for backpropagation
                self._save_mask(x=a_prev_slice, cords=(i, j))
                
                # Perform max pooling
                output[:, i, j, :] = np.max(a_prev_slice, axis=(1, 2))
        
        # Remove batch dimension if input was single image
        if single_image:
            output = output[0]
            
        return output

    def backward(self, da_curr: np.array) -> np.array:
        """
        Backward pass through max pooling
        :param da_curr - gradient from next layer
        :return - gradient w.r.t. input
        """
        # Handle single image case
        single_image = False
        if len(da_curr.shape) == 3:
            single_image = True
            da_curr = da_curr[np.newaxis, ...]
            
        output = np.zeros_like(self._a)
        _, h_out, w_out, _ = da_curr.shape
        h_pool, w_pool = self._pool_size

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self._stride
                h_end = h_start + h_pool
                w_start = j * self._stride
                w_end = w_start + w_pool
                
                # Apply cached mask to distribute gradients
                output[:, h_start:h_end, w_start:w_end, :] += \
                    da_curr[:, i:i + 1, j:j + 1, :] * self._cache.get((i, j), 0)
        
        # Remove batch dimension if input was single image
        if single_image:
            output = output[0]
            
        return output

    def _save_mask(self, x: np.array, cords: Tuple[int, int]) -> None:
        """
        Create and save mask indicating where max values are located
        :param x - input slice (N, h_pool, w_pool, C)
        :param cords - coordinates (i, j) in output feature map
        """
        mask = np.zeros_like(x)
        n, h, w, c = x.shape
        
        # Reshape to find argmax along spatial dimensions
        x_reshaped = x.reshape(n, h * w, c)
        idx = np.argmax(x_reshaped, axis=1)  # Shape: (n, c)

        # Create indices for advanced indexing
        n_idx, c_idx = np.indices((n, c))
        
        # Set mask values to 1 where max occurs
        mask_reshaped = mask.reshape(n, h * w, c)
        mask_reshaped[n_idx, idx, c_idx] = 1
        
        self._cache[cords] = mask