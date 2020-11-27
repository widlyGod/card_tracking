# YOLOv5 NCNN Implementation

This repo provides C++ implementation of [YOLOv4 model] using
Tencent's NCNN framework.

# Notes

Currently NCNN does not support Slice operations with steps, therefore I removed the Slice operation
and replaced the input with a downscaled image and stacked it to match the channel number. This
may slightly reduce the accuracy.

# Credits 

* [NCNN by Tencent](https://github.com/tencent/ncnn) 
