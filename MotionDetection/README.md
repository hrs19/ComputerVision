# Motion Detection Using Image Filtering

This Python script performs motion detection in image sequences captured with a stationary camera using image filtering techniques. The code allows you to explore various options for motion detection and image preprocessing.

## Description

The script provides two main functions:

1. `Gauss_1D(tsigma)`: This function generates a 1D Gaussian kernel with a specified standard deviation (`tsigma`). It is used for smoothing the images.

2. `motion_det_10(folder='RedChair', filter_kernel=np.array([-1, 0, 1]), smoothening=False, derivative='derivative_filter')`: This function performs motion detection on a sequence of image frames.

## Usage

You can use the provided functions for motion detection by customizing the parameters in the script. Here's an example of how to call the functions:

```python
if __name__ == "__main__":
    # Customize the parameters as needed
    filter_kernel = np.array([-0.5, 0, 0.5])
    folder = 'Office'
    derivative = 'derivative_filter'
    mask_append, abs_derivative_append, threshold_val_li = motion_det_10(folder=folder, derivative=derivative, smoothening=False, filter_kernel=filter_kernel)
```

## Parameters

- `folder`: Specify the folder containing the image sequence you want to analyze.
- `filter_kernel`: Define the filter kernel for the temporal derivative. You can choose from various options, including custom kernels and Gaussian kernels.
- `smoothening`: Enable smoothening of the images before applying the derivative filter. You can choose between box filters and Gaussian filters with a custom standard deviation (`ssigma`).
- `derivative`: Choose the type of derivative filter to apply, such as a simple filter or a Gaussian filter with a custom standard deviation (`tsigma`).
