# Function to set up the device
def set_device(device="cpu", idx=0):
    """
    Sets up the device for training or inference.

    This function checks if CUDA is available and sets the device accordingly.
    If CUDA is available, it tries to use the specified GPU. If the specified GPU
    is not available, it defaults to the first GPU. If CUDA is not available, it defaults to CPU.

    Args:
        device (str, optional): Desired device type. Default is "cpu".
        idx (int, optional): Index of the GPU to be used if available. Default is 0.

    Returns:
        torch.device: The device that will be used for training or inference.
    """
    if device != "cpu":
        if torch.cuda.device_count() > idx and torch.cuda.is_available():
            print(
                "Cuda installed! Running on GPU {} {}!".format(
                    idx, torch.cuda.get_device_name(idx)
                )
            )
            device = "cuda:{}".format(idx)
        elif torch.cuda.device_count() > 0 and torch.cuda.is_available():
            print(
                "Cuda installed but only {} GPU(s) available! Running on GPU 0 {}!".format(
                    torch.cuda.device_count(), torch.cuda.get_device_name()
                )
            )
            device = "cuda:0"
        else:
            print("No GPU available! Running on CPU")
    return device
