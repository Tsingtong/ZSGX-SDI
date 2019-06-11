from skimage.measure import compare_mse as mse
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr


def calculateMSE(img1, img2):
    MSE = mse(img1, img2)
    return MSE


def calculatePSNR(img1, img2):
    PSNR = psnr(img1, img2)
    return PSNR


def calculateSSIM(img1, img2):
    SSIM = ssim(img1, img2, multichannel=True)
    return SSIM


def calculateALL(img1, img2):
    MSE = mse(img1, img2)
    PSNR = psnr(img1, img2)
    SSIM = ssim(img1, img2, multichannel=True)
    return MSE, PSNR, SSIM


class Queue:
    def __init__(self):
        self.items = []

    def enqueue(self, item):
        self.items.append(item)

    def dequeue(self):
        return self.items.pop(0)

    def empty(self):
        return self.size() == 0

    def size(self):
        return len(self.items)

    def proDuplicate(self):
        """
        :rtype: counter
        """
        duplicate_counter = 0
        consistent = True
        for index, value in enumerate(self.items):
            if index == 0:  # 如果当前元素和当前元素的下标不相同
                continue
            else:
                if not consistent:
                    break
                if value == self.items[index-1]:  # 如果当前元素和前一个元素值相同，说明重复
                    duplicate_counter += 1
                else:
                    consistent = False
        return duplicate_counter+1

    def folDuplicate(self):
        """
        :rtype: counter
        """
        duplicate_counter = 0
        consistent = True
        for index, value in enumerate(reversed(self.items)):
            if index == 0:  # 如果当前元素和当前元素的下标不相同
                continue
            else:
                if not consistent:
                    break
                if value == self.items[index-1]:  # 如果当前元素和前一个元素值相同，说明重复
                    duplicate_counter += 1
                else:
                    consistent = False
        return duplicate_counter+1
