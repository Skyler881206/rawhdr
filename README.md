# RawHDR: High Dynamic Range Image Reconstruction from a Single Raw Image

Paper address: https://arxiv.org/abs/2309.02020

This code is designed to rebuild the rawhdr model using an LDR-HDR image dataset.

The original model converts 14-bit raw files to 20-bit HDR images. However, this GitHub repository generates 32-bit HDR images from 8-bit LDR inputs. The input images are in the sRGB domain, which differs from the original training dataset.

To enable the model to use sRGB LDR images as input, the provided code transforms the sRGB images into a 4-channel RGBG format by re-concatenating the input LDR images.