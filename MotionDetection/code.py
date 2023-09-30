import os
import cv2
import numpy as np
import scipy.signal as signal

def Gauss_1D(tsigma):
    length = int(5 * tsigma)
    gaussian = signal.gaussian(length, std=tsigma)
    min_y = min(gaussian)
    for i in range(length):
        gaussian[i] = np.round(gaussian[i] / min_y)
    print('(1/', sum(gaussian), ')*', gaussian)
    return gaussian / sum(gaussian)

def motion_det_10(folder='RedChair', filter_kernel=np.array([-1, 0, 1]), smoothening=False, derivative='derivative_filter'):
    folder = f'Data/{folder}'
    dir_list = os.listdir(f'{folder}')
    if smoothening:
        smoothening = smoothening.split('_')
        size = int(smoothening[1])
        if smoothening[0] == 'gauss':
            ssigma = input('ssigma = ')
        print(smoothening)
    if derivative == 'gauss':
        tsigma = float(input('tsigma = '))
        filter_kernel = Gauss_1D(tsigma)
        print('kernel=', filter_kernel)

    abs_derivative_append = []
    mask_append = []
    buffer = []
    threshold_val_li = []

    threshold_value = 10

    for im in dir_list:
        img = cv2.imread(f'{folder}/{im}')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if smoothening:
            if smoothening[0] == 'box':
                gray = cv2.boxFilter(gray, ddepth=0, ksize=(size, size))

            elif smoothening[0] == 'gauss':
                l = int(5 * ssigma[0])
                gray = cv2.GaussianBlur(gray, (0, 0), int(ssigma), int(ssigma))

        buffer.append(gray)

        if len(buffer) >= 3:
            derivative_0 = cv2.filter2D(buffer[0], -1, filter_kernel)
            derivative_1 = cv2.filter2D(buffer[2], -1, filter_kernel)

            abs_derivative_x = cv2.absdiff(derivative_0, derivative_1)

            std_abs_der = np.std(abs_derivative_x)
            threshold_value = 10

            threshold_val_li.append(threshold_value)

            _, mask = cv2.threshold(abs_derivative_x, threshold_value, 255, cv2.THRESH_BINARY)

            masked_image = np.multiply(mask, buffer[1])
            gray = np.hstack((buffer[1], masked_image, abs_derivative_x))

            abs_derivative_append.append(abs_derivative_x)
            mask_append.append(masked_image)
            try:
                cv2.imwrite(f'E:/Course Work Northeastern/ComputerVision/Project/Project1/Code/{folder}_result/{im}_res.jpg', gray)
            except Exception as ex:
                print(ex)

            buffer.pop(0)
            cv2.imshow(f'Gray', gray)
            cv2.imshow('img', img)
            k = cv2.waitKey(40) & 0xFF
            if k == 27 or k == ord('q') or k == ord('Q'):
                cv2.destroyAllWindows()
                break

    cv2.destroyAllWindows()
    return mask_append, abs_derivative_append, threshold_val_li

def motion_det(folder='RedChair', filter_kernel=np.array([-1, 0, 1]), smoothening=False, derivative='derivative_filter'):
    folder = f'Data/{folder}'
    dir_list = os.listdir(f'{folder}')
    if smoothening:
        smoothening = smoothening.split('_')
        size = int(smoothening[1])
        if smoothening[0] == 'gauss':
            ssigma = input('ssigma = ')
        print(smoothening)
    if derivative == 'gauss':
        tsigma = float(input('tsigma = '))
        filter_kernel = Gauss_1D(tsigma)
        print('kernel=', filter_kernel)

    abs_derivative_append = []
    mask_append = []
    buffer = []
    threshold_val_li = []

    threshold_value = 10

    for im in dir_list:
        img = cv2.imread(f'{folder}/{im}')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if smoothening:
            if smoothening[0] == 'box':
                gray = cv2.boxFilter(gray, ddepth=0, ksize=(size, size))

            elif smoothening[0] == 'gauss':
                l = int(5 * ssigma[0])
                gray = cv2.GaussianBlur(gray, (0, 0), int(ssigma), int(ssigma))

        buffer.append(gray)

        if len(buffer) >= 3:
            derivative_0 = cv2.filter2D(buffer[0], -1, filter_kernel)
            derivative_1 = cv2.filter2D(buffer[2], -1, filter_kernel)

            abs_derivative_x = cv2.absdiff(derivative_0, derivative_1)

            std_abs_der = np.std(abs_derivative_x)
            temp_max_int = 0
            if derivative == 'derivative_filter':
                if std_abs_der < 1.0:
                    temp_max_int = abs_derivative_x.max()
                    threshold_value = temp_max_int * 1.1
                else:
                    threshold_value = std_abs_der * 5

            else:
                if std_abs_der < 1.0:
                    temp_max_int = abs_derivative_x.max()
                    threshold_value = temp_max_int * 1.1

                else:
                    threshold_value = cv2.THRESH_OTSU

            threshold_val_li.append(threshold_value)

            _, mask = cv2.threshold(abs_derivative_x, threshold_value, 255, cv2.THRESH_BINARY)

            masked_image = np.multiply(mask, buffer[1])
            gray = np.hstack((buffer[1], masked_image, abs_derivative_x))

            abs_derivative_append.append(abs_derivative_x)
            mask_append.append(masked_image)
            try:
                cv2.imwrite(f'E:/Course Work Northeastern/ComputerVision/Project/Project1/Code/{folder}_result/{im}_res.jpg', gray)
            except Exception as ex:
                print(ex)

            buffer.pop(0)
            cv2.imshow(f'Gray', gray)
            cv2.imshow('img', img)
            k = cv2.waitKey(40) & 0xFF
            if k == 27 or k == ord('q') or k == ord('Q'):
                cv2.destroyAllWindows()
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Call your functions with desired parameters here
    filter_kernel = np.array([-0.5, 0, 0.5])
    folder = 'Office'
    derivative = 'derivative_filter'
    mask_append, abs_derivative_append, threshold_val_li = motion_det_10(folder=folder, derivative=derivative, smoothening=False, filter_kernel=filter_kernel)
