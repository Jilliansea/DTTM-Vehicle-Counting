import os
import sys

def frame_preprocess(img, w, h , w_stride, h_stride):
    W_BLOCK_NUM = int(w / w_stride)
    H_BLOCK_NUM = int(h / h_stride)

    reshape_img = img.reshape(H_BLOCK_NUM, h_stride, W_BLOCK_NUM, w_stride, 3)
    reshape_img_mean = reshape_img.mean(axis=(1, 3)).astype('uint8')
    return reshape_img, reshape_img_mean
