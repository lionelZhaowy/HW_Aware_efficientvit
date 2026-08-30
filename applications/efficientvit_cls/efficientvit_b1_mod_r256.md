相比b1版标准模型的修改：
- 输出改为100类
- Backbone的激活函数hswish改为relu、DSConv的激活函数relu6改为relu
- input_stem第1各ConvLayer之后添加DSConv
- build_local_block的expand_ratio从1改为2
- ClsHead的通道改为[768, 512]，Linear层从ln归一化改为bn1d归一化
- LiteMLA：
  - QKV计算分开
  - 移除注意力归一化
  - 动态维度计算移除
- 数据集Normalize参数改为mean和std都是0.5
- 添加MQBench量化代码


EfficientViTCls(
  (backbone): EfficientViTBackbone(
    (input_stem): OpSequential(
      (op_list): ModuleList(
        (0): ConvLayer(
          (conv): Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (norm): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): ReLU()
        )
        (1): DSConv(
          (depth_conv): ConvLayer(
            (conv): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
            (norm): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): ReLU()
          )
          (point_conv): ConvLayer(
            (conv): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (2): ResidualBlock(
          (main): MBConv(
            (inverted_conv): ConvLayer(
              (conv): Conv2d(16, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (norm): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act): ReLU()
            )
            (depth_conv): ConvLayer(
              (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
              (norm): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act): ReLU()
            )
            (point_conv): ConvLayer(
              (conv): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (norm): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (shortcut): IdentityLayer()
        )
      )
    )
    (stages): ModuleList(
      (0): OpSequential(
        (op_list): ModuleList(
          (0): ResidualBlock(
            (main): MBConv(
              (inverted_conv): ConvLayer(
                (conv): Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): ReLU()
              )
              (depth_conv): ConvLayer(
                (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=64, bias=False)
                (norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): ReLU()
              )
              (point_conv): ConvLayer(
                (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (norm): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
          )
          (1): ResidualBlock(
            (main): MBConv(
              (inverted_conv): ConvLayer(
                (conv): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (norm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): ReLU()
              )
              (depth_conv): ConvLayer(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
                (norm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): ReLU()
              )
              (point_conv): ConvLayer(
                (conv): Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (norm): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (shortcut): IdentityLayer()
          )
        )
      )
      (1): OpSequential(
        (op_list): ModuleList(
          (0): ResidualBlock(
            (main): MBConv(
              (inverted_conv): ConvLayer(
                (conv): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (norm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): ReLU()
              )
              (depth_conv): ConvLayer(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=128, bias=False)
                (norm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): ReLU()
              )
              (point_conv): ConvLayer(
                (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
          )
          (1-2): 2 x ResidualBlock(
            (main): MBConv(
              (inverted_conv): ConvLayer(
                (conv): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (norm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): ReLU()
              )
              (depth_conv): ConvLayer(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
                (norm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): ReLU()
              )
              (point_conv): ConvLayer(
                (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (shortcut): IdentityLayer()
          )
        )
      )
      (2): OpSequential(
        (op_list): ModuleList(
          (0): ResidualBlock(
            (main): MBConv(
              (inverted_conv): ConvLayer(
                (conv): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
                (act): ReLU()
              )
              (depth_conv): ConvLayer(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
                (act): ReLU()
              )
              (point_conv): ConvLayer(
                (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (norm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
          )
          (1-3): 3 x EfficientViTBlock(
            (context_module): ResidualBlock(
              (main): LiteMLA(
                (qkv): ConvLayer(
                  (conv): Conv2d(128, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (aggreg): ModuleList(
                  (0): Sequential(
                    (0): Conv2d(384, 384, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=384, bias=False)
                    (1): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), groups=24, bias=False)
                  )
                )
                (kernel_func): ReLU()
                (proj): ConvLayer(
                  (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (norm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
              )
              (shortcut): IdentityLayer()
            )
            (local_module): ResidualBlock(
              (main): MBConv(
                (inverted_conv): ConvLayer(
                  (conv): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1))
                  (act): ReLU()
                )
                (depth_conv): ConvLayer(
                  (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512)
                  (act): ReLU()
                )
                (point_conv): ConvLayer(
                  (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (norm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
              )
              (shortcut): IdentityLayer()
            )
          )
        )
      )
      (3): OpSequential(
        (op_list): ModuleList(
          (0): ResidualBlock(
            (main): MBConv(
              (inverted_conv): ConvLayer(
                (conv): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1))
                (act): ReLU()
              )
              (depth_conv): ConvLayer(
                (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
                (act): ReLU()
              )
              (point_conv): ConvLayer(
                (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (norm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
          )
          (1-4): 4 x EfficientViTBlock(
            (context_module): ResidualBlock(
              (main): LiteMLA(
                (qkv): ConvLayer(
                  (conv): Conv2d(256, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (aggreg): ModuleList(
                  (0): Sequential(
                    (0): Conv2d(768, 768, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=768, bias=False)
                    (1): Conv2d(768, 768, kernel_size=(1, 1), stride=(1, 1), groups=48, bias=False)
                  )
                )
                (kernel_func): ReLU()
                (proj): ConvLayer(
                  (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (norm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
              )
              (shortcut): IdentityLayer()
            )
            (local_module): ResidualBlock(
              (main): MBConv(
                (inverted_conv): ConvLayer(
                  (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1))
                  (act): ReLU()
                )
                (depth_conv): ConvLayer(
                  (conv): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),groups=1024)
                  (act): ReLU()
                )
                (point_conv): ConvLayer(
                  (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (norm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
              )
              (shortcut): IdentityLayer()
            )
          )
        )
      )
    )
  )
  (head): ClsHead(
    (op_list): ModuleList(
      (0): ConvLayer(
        (conv): Conv2d(256, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (norm): BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act): Hardswish()
      )
      (1): AdaptiveAvgPool2d(output_size=1)
      (2): LinearLayer(
        (linear): Linear(in_features=768, out_features=512, bias=False)
        (act): Hardswish()
      )
      (3): LinearLayer(
        (linear): Linear(in_features=512, out_features=100, bias=True)
      )
    )
  )
)
------------------------------Happy every day! :)---------------------------------
-----------------------------Author: Peiyi & Ping---------------------------------
        Layer (type)         Output Shape      O-Size(MB)       Param #      P-Size(MB)
==================================================================================
            Conv2d-1     [1, 16, 128, 128]     1.0 MB           432 0.00164794921875 MB
       BatchNorm2d-2     [1, 16, 128, 128]     1.0 MB            32 0.0001220703125 MB
              ReLU-3     [1, 16, 128, 128]     1.0 MB             0          0.0 MB
            Conv2d-4     [1, 16, 128, 128]     1.0 MB           144 0.00054931640625 MB
       BatchNorm2d-5     [1, 16, 128, 128]     1.0 MB            32 0.0001220703125 MB
              ReLU-6     [1, 16, 128, 128]     1.0 MB             0          0.0 MB
            Conv2d-7     [1, 16, 128, 128]     1.0 MB           256 0.0009765625 MB
       BatchNorm2d-8     [1, 16, 128, 128]     1.0 MB            32 0.0001220703125 MB
            Conv2d-9     [1, 32, 128, 128]     2.0 MB           512  0.001953125 MB
      BatchNorm2d-10     [1, 32, 128, 128]     2.0 MB            64 0.000244140625 MB
             ReLU-11     [1, 32, 128, 128]     2.0 MB             0          0.0 MB
           Conv2d-12     [1, 32, 128, 128]     2.0 MB           288 0.0010986328125 MB
      BatchNorm2d-13     [1, 32, 128, 128]     2.0 MB            64 0.000244140625 MB
             ReLU-14     [1, 32, 128, 128]     2.0 MB             0          0.0 MB
           Conv2d-15     [1, 16, 128, 128]     1.0 MB           512  0.001953125 MB
      BatchNorm2d-16     [1, 16, 128, 128]     1.0 MB            32 0.0001220703125 MB
    IdentityLayer-17     [1, 16, 128, 128]     1.0 MB             0          0.0 MB
           Conv2d-18     [1, 64, 128, 128]     4.0 MB         1,024   0.00390625 MB
      BatchNorm2d-19     [1, 64, 128, 128]     4.0 MB           128 0.00048828125 MB
             ReLU-20     [1, 64, 128, 128]     4.0 MB             0          0.0 MB
           Conv2d-21       [1, 64, 64, 64]     1.0 MB           576 0.002197265625 MB
      BatchNorm2d-22       [1, 64, 64, 64]     1.0 MB           128 0.00048828125 MB
             ReLU-23       [1, 64, 64, 64]     1.0 MB             0          0.0 MB
           Conv2d-24       [1, 32, 64, 64]     0.5 MB         2,048    0.0078125 MB
      BatchNorm2d-25       [1, 32, 64, 64]     0.5 MB            64 0.000244140625 MB
           Conv2d-26      [1, 128, 64, 64]     2.0 MB         4,096     0.015625 MB
      BatchNorm2d-27      [1, 128, 64, 64]     2.0 MB           256 0.0009765625 MB
             ReLU-28      [1, 128, 64, 64]     2.0 MB             0          0.0 MB
           Conv2d-29      [1, 128, 64, 64]     2.0 MB         1,152 0.00439453125 MB
      BatchNorm2d-30      [1, 128, 64, 64]     2.0 MB           256 0.0009765625 MB
             ReLU-31      [1, 128, 64, 64]     2.0 MB             0          0.0 MB
           Conv2d-32       [1, 32, 64, 64]     0.5 MB         4,096     0.015625 MB
      BatchNorm2d-33       [1, 32, 64, 64]     0.5 MB            64 0.000244140625 MB
    IdentityLayer-34       [1, 32, 64, 64]     0.5 MB             0          0.0 MB
           Conv2d-35      [1, 128, 64, 64]     2.0 MB         4,096     0.015625 MB
      BatchNorm2d-36      [1, 128, 64, 64]     2.0 MB           256 0.0009765625 MB
             ReLU-37      [1, 128, 64, 64]     2.0 MB             0          0.0 MB
           Conv2d-38      [1, 128, 32, 32]     0.5 MB         1,152 0.00439453125 MB
      BatchNorm2d-39      [1, 128, 32, 32]     0.5 MB           256 0.0009765625 MB
             ReLU-40      [1, 128, 32, 32]     0.5 MB             0          0.0 MB
           Conv2d-41       [1, 64, 32, 32]    0.25 MB         8,192      0.03125 MB
      BatchNorm2d-42       [1, 64, 32, 32]    0.25 MB           128 0.00048828125 MB
           Conv2d-43      [1, 256, 32, 32]     1.0 MB        16,384       0.0625 MB
      BatchNorm2d-44      [1, 256, 32, 32]     1.0 MB           512  0.001953125 MB
             ReLU-45      [1, 256, 32, 32]     1.0 MB             0          0.0 MB
           Conv2d-46      [1, 256, 32, 32]     1.0 MB         2,304 0.0087890625 MB
      BatchNorm2d-47      [1, 256, 32, 32]     1.0 MB           512  0.001953125 MB
             ReLU-48      [1, 256, 32, 32]     1.0 MB             0          0.0 MB
           Conv2d-49       [1, 64, 32, 32]    0.25 MB        16,384       0.0625 MB
      BatchNorm2d-50       [1, 64, 32, 32]    0.25 MB           128 0.00048828125 MB
    IdentityLayer-51       [1, 64, 32, 32]    0.25 MB             0          0.0 MB
           Conv2d-52      [1, 256, 32, 32]     1.0 MB        16,384       0.0625 MB
      BatchNorm2d-53      [1, 256, 32, 32]     1.0 MB           512  0.001953125 MB
             ReLU-54      [1, 256, 32, 32]     1.0 MB             0          0.0 MB
           Conv2d-55      [1, 256, 32, 32]     1.0 MB         2,304 0.0087890625 MB
      BatchNorm2d-56      [1, 256, 32, 32]     1.0 MB           512  0.001953125 MB
             ReLU-57      [1, 256, 32, 32]     1.0 MB             0          0.0 MB
           Conv2d-58       [1, 64, 32, 32]    0.25 MB        16,384       0.0625 MB
      BatchNorm2d-59       [1, 64, 32, 32]    0.25 MB           128 0.00048828125 MB
    IdentityLayer-60       [1, 64, 32, 32]    0.25 MB             0          0.0 MB
           Conv2d-61      [1, 256, 32, 32]     1.0 MB        16,640 0.0634765625 MB
             ReLU-62      [1, 256, 32, 32]     1.0 MB             0          0.0 MB
           Conv2d-63      [1, 256, 16, 16]    0.25 MB         2,560  0.009765625 MB
             ReLU-64      [1, 256, 16, 16]    0.25 MB             0          0.0 MB
           Conv2d-65      [1, 128, 16, 16]   0.125 MB        32,768        0.125 MB
      BatchNorm2d-66      [1, 128, 16, 16]   0.125 MB           256 0.0009765625 MB
           Conv2d-67      [1, 384, 16, 16]   0.375 MB        49,152       0.1875 MB
           Conv2d-68      [1, 384, 16, 16]   0.375 MB         9,600 0.03662109375 MB
           Conv2d-69      [1, 384, 16, 16]   0.375 MB         6,144    0.0234375 MB
             ReLU-70      [1, 16, 16, 256]    0.25 MB             0          0.0 MB
             ReLU-71      [1, 16, 16, 256]    0.25 MB             0          0.0 MB
           Conv2d-72      [1, 128, 16, 16]   0.125 MB        32,768        0.125 MB
      BatchNorm2d-73      [1, 128, 16, 16]   0.125 MB           256 0.0009765625 MB
    IdentityLayer-74      [1, 128, 16, 16]   0.125 MB             0          0.0 MB
           Conv2d-75      [1, 512, 16, 16]     0.5 MB        66,048  0.251953125 MB
             ReLU-76      [1, 512, 16, 16]     0.5 MB             0          0.0 MB
           Conv2d-77      [1, 512, 16, 16]     0.5 MB         5,120   0.01953125 MB
             ReLU-78      [1, 512, 16, 16]     0.5 MB             0          0.0 MB
           Conv2d-79      [1, 128, 16, 16]   0.125 MB        65,536         0.25 MB
      BatchNorm2d-80      [1, 128, 16, 16]   0.125 MB           256 0.0009765625 MB
    IdentityLayer-81      [1, 128, 16, 16]   0.125 MB             0          0.0 MB
           Conv2d-82      [1, 384, 16, 16]   0.375 MB        49,152       0.1875 MB
           Conv2d-83      [1, 384, 16, 16]   0.375 MB         9,600 0.03662109375 MB
           Conv2d-84      [1, 384, 16, 16]   0.375 MB         6,144    0.0234375 MB
             ReLU-85      [1, 16, 16, 256]    0.25 MB             0          0.0 MB
             ReLU-86      [1, 16, 16, 256]    0.25 MB             0          0.0 MB
           Conv2d-87      [1, 128, 16, 16]   0.125 MB        32,768        0.125 MB
      BatchNorm2d-88      [1, 128, 16, 16]   0.125 MB           256 0.0009765625 MB
    IdentityLayer-89      [1, 128, 16, 16]   0.125 MB             0          0.0 MB
           Conv2d-90      [1, 512, 16, 16]     0.5 MB        66,048  0.251953125 MB
             ReLU-91      [1, 512, 16, 16]     0.5 MB             0          0.0 MB
           Conv2d-92      [1, 512, 16, 16]     0.5 MB         5,120   0.01953125 MB
             ReLU-93      [1, 512, 16, 16]     0.5 MB             0          0.0 MB
           Conv2d-94      [1, 128, 16, 16]   0.125 MB        65,536         0.25 MB
      BatchNorm2d-95      [1, 128, 16, 16]   0.125 MB           256 0.0009765625 MB
    IdentityLayer-96      [1, 128, 16, 16]   0.125 MB             0          0.0 MB
           Conv2d-97      [1, 384, 16, 16]   0.375 MB        49,152       0.1875 MB
           Conv2d-98      [1, 384, 16, 16]   0.375 MB         9,600 0.03662109375 MB
           Conv2d-99      [1, 384, 16, 16]   0.375 MB         6,144    0.0234375 MB
            ReLU-100      [1, 16, 16, 256]    0.25 MB             0          0.0 MB
            ReLU-101      [1, 16, 16, 256]    0.25 MB             0          0.0 MB
          Conv2d-102      [1, 128, 16, 16]   0.125 MB        32,768        0.125 MB
     BatchNorm2d-103      [1, 128, 16, 16]   0.125 MB           256 0.0009765625 MB
   IdentityLayer-104      [1, 128, 16, 16]   0.125 MB             0          0.0 MB
          Conv2d-105      [1, 512, 16, 16]     0.5 MB        66,048  0.251953125 MB
            ReLU-106      [1, 512, 16, 16]     0.5 MB             0          0.0 MB
          Conv2d-107      [1, 512, 16, 16]     0.5 MB         5,120   0.01953125 MB
            ReLU-108      [1, 512, 16, 16]     0.5 MB             0          0.0 MB
          Conv2d-109      [1, 128, 16, 16]   0.125 MB        65,536         0.25 MB
     BatchNorm2d-110      [1, 128, 16, 16]   0.125 MB           256 0.0009765625 MB
   IdentityLayer-111      [1, 128, 16, 16]   0.125 MB             0          0.0 MB
          Conv2d-112      [1, 512, 16, 16]     0.5 MB        66,048  0.251953125 MB
            ReLU-113      [1, 512, 16, 16]     0.5 MB             0          0.0 MB
          Conv2d-114        [1, 512, 8, 8]   0.125 MB         5,120   0.01953125 MB
            ReLU-115        [1, 512, 8, 8]   0.125 MB             0          0.0 MB
          Conv2d-116        [1, 256, 8, 8]  0.0625 MB       131,072          0.5 MB
     BatchNorm2d-117        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
          Conv2d-118        [1, 768, 8, 8]  0.1875 MB       196,608         0.75 MB
          Conv2d-119        [1, 768, 8, 8]  0.1875 MB        19,200 0.0732421875 MB
          Conv2d-120        [1, 768, 8, 8]  0.1875 MB        12,288     0.046875 MB
            ReLU-121       [1, 32, 16, 64]   0.125 MB             0          0.0 MB
            ReLU-122       [1, 32, 16, 64]   0.125 MB             0          0.0 MB
          Conv2d-123        [1, 256, 8, 8]  0.0625 MB       131,072          0.5 MB
     BatchNorm2d-124        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
   IdentityLayer-125        [1, 256, 8, 8]  0.0625 MB             0          0.0 MB
          Conv2d-126       [1, 1024, 8, 8]    0.25 MB       263,168   1.00390625 MB
            ReLU-127       [1, 1024, 8, 8]    0.25 MB             0          0.0 MB
          Conv2d-128       [1, 1024, 8, 8]    0.25 MB        10,240    0.0390625 MB
            ReLU-129       [1, 1024, 8, 8]    0.25 MB             0          0.0 MB
          Conv2d-130        [1, 256, 8, 8]  0.0625 MB       262,144          1.0 MB
     BatchNorm2d-131        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
   IdentityLayer-132        [1, 256, 8, 8]  0.0625 MB             0          0.0 MB
          Conv2d-133        [1, 768, 8, 8]  0.1875 MB       196,608         0.75 MB
          Conv2d-134        [1, 768, 8, 8]  0.1875 MB        19,200 0.0732421875 MB
          Conv2d-135        [1, 768, 8, 8]  0.1875 MB        12,288     0.046875 MB
            ReLU-136       [1, 32, 16, 64]   0.125 MB             0          0.0 MB
            ReLU-137       [1, 32, 16, 64]   0.125 MB             0          0.0 MB
          Conv2d-138        [1, 256, 8, 8]  0.0625 MB       131,072          0.5 MB
     BatchNorm2d-139        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
   IdentityLayer-140        [1, 256, 8, 8]  0.0625 MB             0          0.0 MB
          Conv2d-141       [1, 1024, 8, 8]    0.25 MB       263,168   1.00390625 MB
            ReLU-142       [1, 1024, 8, 8]    0.25 MB             0          0.0 MB
          Conv2d-143       [1, 1024, 8, 8]    0.25 MB        10,240    0.0390625 MB
            ReLU-144       [1, 1024, 8, 8]    0.25 MB             0          0.0 MB
          Conv2d-145        [1, 256, 8, 8]  0.0625 MB       262,144          1.0 MB
     BatchNorm2d-146        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
   IdentityLayer-147        [1, 256, 8, 8]  0.0625 MB             0          0.0 MB
          Conv2d-148        [1, 768, 8, 8]  0.1875 MB       196,608         0.75 MB
          Conv2d-149        [1, 768, 8, 8]  0.1875 MB        19,200 0.0732421875 MB
          Conv2d-150        [1, 768, 8, 8]  0.1875 MB        12,288     0.046875 MB
            ReLU-151       [1, 32, 16, 64]   0.125 MB             0          0.0 MB
            ReLU-152       [1, 32, 16, 64]   0.125 MB             0          0.0 MB
          Conv2d-153        [1, 256, 8, 8]  0.0625 MB       131,072          0.5 MB
     BatchNorm2d-154        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
   IdentityLayer-155        [1, 256, 8, 8]  0.0625 MB             0          0.0 MB
          Conv2d-156       [1, 1024, 8, 8]    0.25 MB       263,168   1.00390625 MB
            ReLU-157       [1, 1024, 8, 8]    0.25 MB             0          0.0 MB
          Conv2d-158       [1, 1024, 8, 8]    0.25 MB        10,240    0.0390625 MB
            ReLU-159       [1, 1024, 8, 8]    0.25 MB             0          0.0 MB
          Conv2d-160        [1, 256, 8, 8]  0.0625 MB       262,144          1.0 MB
     BatchNorm2d-161        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
   IdentityLayer-162        [1, 256, 8, 8]  0.0625 MB             0          0.0 MB
          Conv2d-163        [1, 768, 8, 8]  0.1875 MB       196,608         0.75 MB
          Conv2d-164        [1, 768, 8, 8]  0.1875 MB        19,200 0.0732421875 MB
          Conv2d-165        [1, 768, 8, 8]  0.1875 MB        12,288     0.046875 MB
            ReLU-166       [1, 32, 16, 64]   0.125 MB             0          0.0 MB
            ReLU-167       [1, 32, 16, 64]   0.125 MB             0          0.0 MB
          Conv2d-168        [1, 256, 8, 8]  0.0625 MB       131,072          0.5 MB
     BatchNorm2d-169        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
   IdentityLayer-170        [1, 256, 8, 8]  0.0625 MB             0          0.0 MB
          Conv2d-171       [1, 1024, 8, 8]    0.25 MB       263,168   1.00390625 MB
            ReLU-172       [1, 1024, 8, 8]    0.25 MB             0          0.0 MB
          Conv2d-173       [1, 1024, 8, 8]    0.25 MB        10,240    0.0390625 MB
            ReLU-174       [1, 1024, 8, 8]    0.25 MB             0          0.0 MB
          Conv2d-175        [1, 256, 8, 8]  0.0625 MB       262,144          1.0 MB
     BatchNorm2d-176        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
   IdentityLayer-177        [1, 256, 8, 8]  0.0625 MB             0          0.0 MB
          Conv2d-178        [1, 768, 8, 8]  0.1875 MB       196,608         0.75 MB
     BatchNorm2d-179        [1, 768, 8, 8]  0.1875 MB         1,536  0.005859375 MB
       Hardswish-180        [1, 768, 8, 8]  0.1875 MB             0          0.0 MB
AdaptiveAvgPool2d-181        [1, 768, 1, 1] 0.0029296875 MB             0          0.0 MB
          Linear-182              [1, 512] 0.001953125 MB       393,216          1.5 MB
       Hardswish-183              [1, 512] 0.001953125 MB             0          0.0 MB
          Linear-184              [1, 100] 0.0003814697265625 MB        51,300 0.1956939697265625 MB
================================================================
Total params: 5,288,068
Trainable params: 5,288,068
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.75
Forward/backward pass size (MB): 201.14
Params size (MB): 20.17
Estimated Total Size (MB): 222.06
----------------------------------------------------------------
[INFO] Register count_convNd() for <class 'torch.nn.modules.conv.Conv2d'>.
[INFO] Register count_normalization() for <class 'torch.nn.modules.batchnorm.BatchNorm2d'>.
[INFO] Register zero_ops() for <class 'torch.nn.modules.activation.ReLU'>.
[INFO] Register zero_ops() for <class 'torch.nn.modules.container.Sequential'>.
[INFO] Register count_adap_avgpool() for <class 'torch.nn.modules.pooling.AdaptiveAvgPool2d'>.
[INFO] Register count_linear() for <class 'torch.nn.modules.linear.Linear'>.
运算量：699.305M, 参数量：5.288M



