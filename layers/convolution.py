import numpy as np

class ConvolutionLayer:
    def __init__(self, num_filters, filter_size, stride=1, padding=1):
        """
        Convolution layer optimized for 64x64x3 images
        :param num_filters - number of filters/kernels
        :param filter_size - size of square filter (e.g., 3 for 3x3)
        :param stride - stride for convolution
        :param padding - padding to add around input
        """
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.filters = None
        self.bias = None
        self.last_input = None
        self.last_padded_input = None
        
    def initialize_weights(self, input_channels):
        """
        Initialize filters using He initialization (good for ReLU)
        """
        fan_in = input_channels * self.filter_size * self.filter_size
        
        # He initialization
        self.filters = np.random.randn(
            self.filter_size, self.filter_size, input_channels, self.num_filters
        ) * np.sqrt(2.0 / fan_in)
        
        self.bias = np.zeros((self.num_filters,))
    
    def add_padding(self, input_data):
        """
        Add zero padding around input
        """
        if self.padding == 0:
            return input_data
            
        if len(input_data.shape) == 3:  # Single image (H, W, C)
            return np.pad(input_data, 
                         ((self.padding, self.padding), 
                          (self.padding, self.padding), 
                          (0, 0)), 
                         mode='constant', constant_values=0)
        else:  # Batch (N, H, W, C)
            return np.pad(input_data, 
                         ((0, 0),
                          (self.padding, self.padding), 
                          (self.padding, self.padding), 
                          (0, 0)), 
                         mode='constant', constant_values=0)
    
    def forward(self, input_data):
        """
        Forward pass through convolution layer
        :param input_data - input tensor (H, W, C) or (N, H, W, C)
        :return - convolved output
        """
        self.last_input = np.array(input_data, copy=True)
        
        # Handle single image
        single_image = False
        if len(input_data.shape) == 3:
            single_image = True
            input_data = input_data[np.newaxis, ...]
        
        # Initialize weights if first forward pass
        if self.filters is None:
            self.initialize_weights(input_data.shape[3])
        
        # Add padding
        self.last_padded_input = self.add_padding(input_data)
        N, H, W, C = self.last_padded_input.shape
        
        # Calculate output dimensions
        H_out = (H - self.filter_size) // self.stride + 1
        W_out = (W - self.filter_size) // self.stride + 1
        
        output = np.zeros((N, H_out, W_out, self.num_filters))
        
        # Perform convolution
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * self.stride
                h_end = h_start + self.filter_size
                w_start = j * self.stride
                w_end = w_start + self.filter_size
                
                # Extract patches from input
                patches = self.last_padded_input[:, h_start:h_end, w_start:w_end, :]
                
                # Convolution operation using broadcasting
                conv_result = np.sum(
                    patches[:, :, :, :, np.newaxis] * self.filters[np.newaxis, :, :, :, :],
                    axis=(1, 2, 3)
                )
                
                output[:, i, j, :] = conv_result + self.bias
        
        # Remove batch dimension if input was single image
        if single_image:
            output = output[0]
            
        return output

    def backward(self, dout, learning_rate=0.001):
        """
        Backward pass through convolution layer
        :param dout - gradient from next layer
        :param learning_rate - learning rate for weight updates
        :return - gradient w.r.t. input
        """
        # Handle single image case
        single_image = False
        original_input_shape = self.last_input.shape
        
        if len(dout.shape) == 3:
            single_image = True
            dout = dout[np.newaxis, ...]
            
        if len(self.last_input.shape) == 3:
            padded_input = self.last_padded_input[0:1]  # Keep batch dimension
        else:
            padded_input = self.last_padded_input
            
        N, H_out, W_out, _ = dout.shape
        _, H_pad, W_pad, C = padded_input.shape
        
        # Initialize gradients
        dfilters = np.zeros_like(self.filters)
        dbias = np.sum(dout, axis=(0, 1, 2)) / N
        dx_padded = np.zeros_like(padded_input)
        
        # Compute gradients
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * self.stride
                h_end = h_start + self.filter_size
                w_start = j * self.stride
                w_end = w_start + self.filter_size
                
                # Patches for this output position
                patches = padded_input[:, h_start:h_end, w_start:w_end, :]
                
                # Gradient w.r.t. filters
                dfilters += np.sum(
                    patches[:, :, :, :, np.newaxis] * dout[:, i:i+1, j:j+1, np.newaxis, :],
                    axis=0
                ) / N
                
                # Gradient w.r.t. input
                dx_padded[:, h_start:h_end, w_start:w_end, :] += np.sum(
                    self.filters[np.newaxis, :, :, :, :] * dout[:, i:i+1, j:j+1, np.newaxis, :],
                    axis=4
                )
        
        # Remove padding from gradient
        if self.padding > 0:
            dx = dx_padded[:, self.padding:-self.padding, self.padding:-self.padding, :]
        else:
            dx = dx_padded
        
        # Update weights
        self.filters -= learning_rate * dfilters
        self.bias -= learning_rate * dbias
        
        # Restore original shape
        if single_image:
            dx = dx[0]
            
        return dx

    def get_output_shape(self, input_shape):
        """
        Calculate output shape given input shape
        :param input_shape - (H, W, C) or (N, H, W, C)
        :return - output shape
        """
        if len(input_shape) == 3:
            H, W, C = input_shape
            H_out = (H + 2 * self.padding - self.filter_size) // self.stride + 1
            W_out = (W + 2 * self.padding - self.filter_size) // self.stride + 1
            return (H_out, W_out, self.num_filters)
        else:
            N, H, W, C = input_shape
            H_out = (H + 2 * self.padding - self.filter_size) // self.stride + 1
            W_out = (W + 2 * self.padding - self.filter_size) // self.stride + 1
            return (N, H_out, W_out, self.num_filters)