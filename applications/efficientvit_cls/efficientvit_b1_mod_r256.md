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
                (q_proj): ConvLayer(
                  (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (k_proj): ConvLayer(
                  (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (v_proj): ConvLayer(
                  (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (aggreg_q): ModuleList(
                  (0): Sequential(
                    (0): ConvLayer(
                      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
                      (norm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (1): ConvLayer(
                      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
                      (norm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (2): ConvLayer(
                      (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                      (norm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                  )
                )
                (aggreg_k): ModuleList(
                  (0): Sequential(
                    (0): ConvLayer(
                      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
                      (norm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (1): ConvLayer(
                      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
                      (norm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (2): ConvLayer(
                      (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                      (norm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                  )
                )
                (aggreg_v): ModuleList(
                  (0): Sequential(
                    (0): ConvLayer(
                      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
                      (norm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (1): ConvLayer(
                      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
                      (norm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (2): ConvLayer(
                      (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                      (norm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
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
                (q_proj): ConvLayer(
                  (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (k_proj): ConvLayer(
                  (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (v_proj): ConvLayer(
                  (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (aggreg_q): ModuleList(
                  (0): Sequential(
                    (0): ConvLayer(
                      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
                      (norm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (1): ConvLayer(
                      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
                      (norm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (2): ConvLayer(
                      (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                      (norm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                  )
                )
                (aggreg_k): ModuleList(
                  (0): Sequential(
                    (0): ConvLayer(
                      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
                      (norm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (1): ConvLayer(
                      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
                      (norm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (2): ConvLayer(
                      (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                      (norm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                  )
                )
                (aggreg_v): ModuleList(
                  (0): Sequential(
                    (0): ConvLayer(
                      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
                      (norm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (1): ConvLayer(
                      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
                      (norm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (2): ConvLayer(
                      (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                      (norm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
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
                  (conv): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1024)
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
        (act): ReLU()
      )
      (1): AdaptiveAvgPool2d(output_size=1)
      (2): LinearLayer(
        (linear): Linear(in_features=768, out_features=512, bias=False)
        (norm): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act): ReLU()
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
           Conv2d-67      [1, 128, 16, 16]   0.125 MB        16,384       0.0625 MB
           Conv2d-68      [1, 128, 16, 16]   0.125 MB        16,384       0.0625 MB
           Conv2d-69      [1, 128, 16, 16]   0.125 MB        16,384       0.0625 MB
           Conv2d-70      [1, 128, 16, 16]   0.125 MB         1,152 0.00439453125 MB
      BatchNorm2d-71      [1, 128, 16, 16]   0.125 MB           256 0.0009765625 MB
           Conv2d-72      [1, 128, 16, 16]   0.125 MB         1,152 0.00439453125 MB
      BatchNorm2d-73      [1, 128, 16, 16]   0.125 MB           256 0.0009765625 MB
           Conv2d-74      [1, 128, 16, 16]   0.125 MB        16,384       0.0625 MB
      BatchNorm2d-75      [1, 128, 16, 16]   0.125 MB           256 0.0009765625 MB
           Conv2d-76      [1, 128, 16, 16]   0.125 MB         1,152 0.00439453125 MB
      BatchNorm2d-77      [1, 128, 16, 16]   0.125 MB           256 0.0009765625 MB
           Conv2d-78      [1, 128, 16, 16]   0.125 MB         1,152 0.00439453125 MB
      BatchNorm2d-79      [1, 128, 16, 16]   0.125 MB           256 0.0009765625 MB
           Conv2d-80      [1, 128, 16, 16]   0.125 MB        16,384       0.0625 MB
      BatchNorm2d-81      [1, 128, 16, 16]   0.125 MB           256 0.0009765625 MB
           Conv2d-82      [1, 128, 16, 16]   0.125 MB         1,152 0.00439453125 MB
      BatchNorm2d-83      [1, 128, 16, 16]   0.125 MB           256 0.0009765625 MB
           Conv2d-84      [1, 128, 16, 16]   0.125 MB         1,152 0.00439453125 MB
      BatchNorm2d-85      [1, 128, 16, 16]   0.125 MB           256 0.0009765625 MB
           Conv2d-86      [1, 128, 16, 16]   0.125 MB        16,384       0.0625 MB
      BatchNorm2d-87      [1, 128, 16, 16]   0.125 MB           256 0.0009765625 MB
             ReLU-88       [1, 8, 16, 512]    0.25 MB             0          0.0 MB
             ReLU-89       [1, 8, 16, 512]    0.25 MB             0          0.0 MB
           Conv2d-90      [1, 128, 16, 16]   0.125 MB        32,768        0.125 MB
      BatchNorm2d-91      [1, 128, 16, 16]   0.125 MB           256 0.0009765625 MB
    IdentityLayer-92      [1, 128, 16, 16]   0.125 MB             0          0.0 MB
           Conv2d-93      [1, 512, 16, 16]     0.5 MB        66,048  0.251953125 MB
             ReLU-94      [1, 512, 16, 16]     0.5 MB             0          0.0 MB
           Conv2d-95      [1, 512, 16, 16]     0.5 MB         5,120   0.01953125 MB
             ReLU-96      [1, 512, 16, 16]     0.5 MB             0          0.0 MB
           Conv2d-97      [1, 128, 16, 16]   0.125 MB        65,536         0.25 MB
      BatchNorm2d-98      [1, 128, 16, 16]   0.125 MB           256 0.0009765625 MB
    IdentityLayer-99      [1, 128, 16, 16]   0.125 MB             0          0.0 MB
          Conv2d-100      [1, 128, 16, 16]   0.125 MB        16,384       0.0625 MB
          Conv2d-101      [1, 128, 16, 16]   0.125 MB        16,384       0.0625 MB
          Conv2d-102      [1, 128, 16, 16]   0.125 MB        16,384       0.0625 MB
          Conv2d-103      [1, 128, 16, 16]   0.125 MB         1,152 0.00439453125 MB
     BatchNorm2d-104      [1, 128, 16, 16]   0.125 MB           256 0.0009765625 MB
          Conv2d-105      [1, 128, 16, 16]   0.125 MB         1,152 0.00439453125 MB
     BatchNorm2d-106      [1, 128, 16, 16]   0.125 MB           256 0.0009765625 MB
          Conv2d-107      [1, 128, 16, 16]   0.125 MB        16,384       0.0625 MB
     BatchNorm2d-108      [1, 128, 16, 16]   0.125 MB           256 0.0009765625 MB
          Conv2d-109      [1, 128, 16, 16]   0.125 MB         1,152 0.00439453125 MB
     BatchNorm2d-110      [1, 128, 16, 16]   0.125 MB           256 0.0009765625 MB
          Conv2d-111      [1, 128, 16, 16]   0.125 MB         1,152 0.00439453125 MB
     BatchNorm2d-112      [1, 128, 16, 16]   0.125 MB           256 0.0009765625 MB
          Conv2d-113      [1, 128, 16, 16]   0.125 MB        16,384       0.0625 MB
     BatchNorm2d-114      [1, 128, 16, 16]   0.125 MB           256 0.0009765625 MB
          Conv2d-115      [1, 128, 16, 16]   0.125 MB         1,152 0.00439453125 MB
     BatchNorm2d-116      [1, 128, 16, 16]   0.125 MB           256 0.0009765625 MB
          Conv2d-117      [1, 128, 16, 16]   0.125 MB         1,152 0.00439453125 MB
     BatchNorm2d-118      [1, 128, 16, 16]   0.125 MB           256 0.0009765625 MB
          Conv2d-119      [1, 128, 16, 16]   0.125 MB        16,384       0.0625 MB
     BatchNorm2d-120      [1, 128, 16, 16]   0.125 MB           256 0.0009765625 MB
            ReLU-121       [1, 8, 16, 512]    0.25 MB             0          0.0 MB
            ReLU-122       [1, 8, 16, 512]    0.25 MB             0          0.0 MB
          Conv2d-123      [1, 128, 16, 16]   0.125 MB        32,768        0.125 MB
     BatchNorm2d-124      [1, 128, 16, 16]   0.125 MB           256 0.0009765625 MB
   IdentityLayer-125      [1, 128, 16, 16]   0.125 MB             0          0.0 MB
          Conv2d-126      [1, 512, 16, 16]     0.5 MB        66,048  0.251953125 MB
            ReLU-127      [1, 512, 16, 16]     0.5 MB             0          0.0 MB
          Conv2d-128      [1, 512, 16, 16]     0.5 MB         5,120   0.01953125 MB
            ReLU-129      [1, 512, 16, 16]     0.5 MB             0          0.0 MB
          Conv2d-130      [1, 128, 16, 16]   0.125 MB        65,536         0.25 MB
     BatchNorm2d-131      [1, 128, 16, 16]   0.125 MB           256 0.0009765625 MB
   IdentityLayer-132      [1, 128, 16, 16]   0.125 MB             0          0.0 MB
          Conv2d-133      [1, 128, 16, 16]   0.125 MB        16,384       0.0625 MB
          Conv2d-134      [1, 128, 16, 16]   0.125 MB        16,384       0.0625 MB
          Conv2d-135      [1, 128, 16, 16]   0.125 MB        16,384       0.0625 MB
          Conv2d-136      [1, 128, 16, 16]   0.125 MB         1,152 0.00439453125 MB
     BatchNorm2d-137      [1, 128, 16, 16]   0.125 MB           256 0.0009765625 MB
          Conv2d-138      [1, 128, 16, 16]   0.125 MB         1,152 0.00439453125 MB
     BatchNorm2d-139      [1, 128, 16, 16]   0.125 MB           256 0.0009765625 MB
          Conv2d-140      [1, 128, 16, 16]   0.125 MB        16,384       0.0625 MB
     BatchNorm2d-141      [1, 128, 16, 16]   0.125 MB           256 0.0009765625 MB
          Conv2d-142      [1, 128, 16, 16]   0.125 MB         1,152 0.00439453125 MB
     BatchNorm2d-143      [1, 128, 16, 16]   0.125 MB           256 0.0009765625 MB
          Conv2d-144      [1, 128, 16, 16]   0.125 MB         1,152 0.00439453125 MB
     BatchNorm2d-145      [1, 128, 16, 16]   0.125 MB           256 0.0009765625 MB
          Conv2d-146      [1, 128, 16, 16]   0.125 MB        16,384       0.0625 MB
     BatchNorm2d-147      [1, 128, 16, 16]   0.125 MB           256 0.0009765625 MB
          Conv2d-148      [1, 128, 16, 16]   0.125 MB         1,152 0.00439453125 MB
     BatchNorm2d-149      [1, 128, 16, 16]   0.125 MB           256 0.0009765625 MB
          Conv2d-150      [1, 128, 16, 16]   0.125 MB         1,152 0.00439453125 MB
     BatchNorm2d-151      [1, 128, 16, 16]   0.125 MB           256 0.0009765625 MB
          Conv2d-152      [1, 128, 16, 16]   0.125 MB        16,384       0.0625 MB
     BatchNorm2d-153      [1, 128, 16, 16]   0.125 MB           256 0.0009765625 MB
            ReLU-154       [1, 8, 16, 512]    0.25 MB             0          0.0 MB
            ReLU-155       [1, 8, 16, 512]    0.25 MB             0          0.0 MB
          Conv2d-156      [1, 128, 16, 16]   0.125 MB        32,768        0.125 MB
     BatchNorm2d-157      [1, 128, 16, 16]   0.125 MB           256 0.0009765625 MB
   IdentityLayer-158      [1, 128, 16, 16]   0.125 MB             0          0.0 MB
          Conv2d-159      [1, 512, 16, 16]     0.5 MB        66,048  0.251953125 MB
            ReLU-160      [1, 512, 16, 16]     0.5 MB             0          0.0 MB
          Conv2d-161      [1, 512, 16, 16]     0.5 MB         5,120   0.01953125 MB
            ReLU-162      [1, 512, 16, 16]     0.5 MB             0          0.0 MB
          Conv2d-163      [1, 128, 16, 16]   0.125 MB        65,536         0.25 MB
     BatchNorm2d-164      [1, 128, 16, 16]   0.125 MB           256 0.0009765625 MB
   IdentityLayer-165      [1, 128, 16, 16]   0.125 MB             0          0.0 MB
          Conv2d-166      [1, 512, 16, 16]     0.5 MB        66,048  0.251953125 MB
            ReLU-167      [1, 512, 16, 16]     0.5 MB             0          0.0 MB
          Conv2d-168        [1, 512, 8, 8]   0.125 MB         5,120   0.01953125 MB
            ReLU-169        [1, 512, 8, 8]   0.125 MB             0          0.0 MB
          Conv2d-170        [1, 256, 8, 8]  0.0625 MB       131,072          0.5 MB
     BatchNorm2d-171        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
          Conv2d-172        [1, 256, 8, 8]  0.0625 MB        65,536         0.25 MB
          Conv2d-173        [1, 256, 8, 8]  0.0625 MB        65,536         0.25 MB
          Conv2d-174        [1, 256, 8, 8]  0.0625 MB        65,536         0.25 MB
          Conv2d-175        [1, 256, 8, 8]  0.0625 MB         2,304 0.0087890625 MB
     BatchNorm2d-176        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
          Conv2d-177        [1, 256, 8, 8]  0.0625 MB         2,304 0.0087890625 MB
     BatchNorm2d-178        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
          Conv2d-179        [1, 256, 8, 8]  0.0625 MB        65,536         0.25 MB
     BatchNorm2d-180        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
          Conv2d-181        [1, 256, 8, 8]  0.0625 MB         2,304 0.0087890625 MB
     BatchNorm2d-182        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
          Conv2d-183        [1, 256, 8, 8]  0.0625 MB         2,304 0.0087890625 MB
     BatchNorm2d-184        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
          Conv2d-185        [1, 256, 8, 8]  0.0625 MB        65,536         0.25 MB
     BatchNorm2d-186        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
          Conv2d-187        [1, 256, 8, 8]  0.0625 MB         2,304 0.0087890625 MB
     BatchNorm2d-188        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
          Conv2d-189        [1, 256, 8, 8]  0.0625 MB         2,304 0.0087890625 MB
     BatchNorm2d-190        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
          Conv2d-191        [1, 256, 8, 8]  0.0625 MB        65,536         0.25 MB
     BatchNorm2d-192        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
            ReLU-193      [1, 16, 16, 128]   0.125 MB             0          0.0 MB
            ReLU-194      [1, 16, 16, 128]   0.125 MB             0          0.0 MB
          Conv2d-195        [1, 256, 8, 8]  0.0625 MB       131,072          0.5 MB
     BatchNorm2d-196        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
   IdentityLayer-197        [1, 256, 8, 8]  0.0625 MB             0          0.0 MB
          Conv2d-198       [1, 1024, 8, 8]    0.25 MB       263,168   1.00390625 MB
            ReLU-199       [1, 1024, 8, 8]    0.25 MB             0          0.0 MB
          Conv2d-200       [1, 1024, 8, 8]    0.25 MB        10,240    0.0390625 MB
            ReLU-201       [1, 1024, 8, 8]    0.25 MB             0          0.0 MB
          Conv2d-202        [1, 256, 8, 8]  0.0625 MB       262,144          1.0 MB
     BatchNorm2d-203        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
   IdentityLayer-204        [1, 256, 8, 8]  0.0625 MB             0          0.0 MB
          Conv2d-205        [1, 256, 8, 8]  0.0625 MB        65,536         0.25 MB
          Conv2d-206        [1, 256, 8, 8]  0.0625 MB        65,536         0.25 MB
          Conv2d-207        [1, 256, 8, 8]  0.0625 MB        65,536         0.25 MB
          Conv2d-208        [1, 256, 8, 8]  0.0625 MB         2,304 0.0087890625 MB
     BatchNorm2d-209        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
          Conv2d-210        [1, 256, 8, 8]  0.0625 MB         2,304 0.0087890625 MB
     BatchNorm2d-211        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
          Conv2d-212        [1, 256, 8, 8]  0.0625 MB        65,536         0.25 MB
     BatchNorm2d-213        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
          Conv2d-214        [1, 256, 8, 8]  0.0625 MB         2,304 0.0087890625 MB
     BatchNorm2d-215        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
          Conv2d-216        [1, 256, 8, 8]  0.0625 MB         2,304 0.0087890625 MB
     BatchNorm2d-217        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
          Conv2d-218        [1, 256, 8, 8]  0.0625 MB        65,536         0.25 MB
     BatchNorm2d-219        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
          Conv2d-220        [1, 256, 8, 8]  0.0625 MB         2,304 0.0087890625 MB
     BatchNorm2d-221        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
          Conv2d-222        [1, 256, 8, 8]  0.0625 MB         2,304 0.0087890625 MB
     BatchNorm2d-223        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
          Conv2d-224        [1, 256, 8, 8]  0.0625 MB        65,536         0.25 MB
     BatchNorm2d-225        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
            ReLU-226      [1, 16, 16, 128]   0.125 MB             0          0.0 MB
            ReLU-227      [1, 16, 16, 128]   0.125 MB             0          0.0 MB
          Conv2d-228        [1, 256, 8, 8]  0.0625 MB       131,072          0.5 MB
     BatchNorm2d-229        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
   IdentityLayer-230        [1, 256, 8, 8]  0.0625 MB             0          0.0 MB
          Conv2d-231       [1, 1024, 8, 8]    0.25 MB       263,168   1.00390625 MB
            ReLU-232       [1, 1024, 8, 8]    0.25 MB             0          0.0 MB
          Conv2d-233       [1, 1024, 8, 8]    0.25 MB        10,240    0.0390625 MB
            ReLU-234       [1, 1024, 8, 8]    0.25 MB             0          0.0 MB
          Conv2d-235        [1, 256, 8, 8]  0.0625 MB       262,144          1.0 MB
     BatchNorm2d-236        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
   IdentityLayer-237        [1, 256, 8, 8]  0.0625 MB             0          0.0 MB
          Conv2d-238        [1, 256, 8, 8]  0.0625 MB        65,536         0.25 MB
          Conv2d-239        [1, 256, 8, 8]  0.0625 MB        65,536         0.25 MB
          Conv2d-240        [1, 256, 8, 8]  0.0625 MB        65,536         0.25 MB
          Conv2d-241        [1, 256, 8, 8]  0.0625 MB         2,304 0.0087890625 MB
     BatchNorm2d-242        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
          Conv2d-243        [1, 256, 8, 8]  0.0625 MB         2,304 0.0087890625 MB
     BatchNorm2d-244        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
          Conv2d-245        [1, 256, 8, 8]  0.0625 MB        65,536         0.25 MB
     BatchNorm2d-246        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
          Conv2d-247        [1, 256, 8, 8]  0.0625 MB         2,304 0.0087890625 MB
     BatchNorm2d-248        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
          Conv2d-249        [1, 256, 8, 8]  0.0625 MB         2,304 0.0087890625 MB
     BatchNorm2d-250        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
          Conv2d-251        [1, 256, 8, 8]  0.0625 MB        65,536         0.25 MB
     BatchNorm2d-252        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
          Conv2d-253        [1, 256, 8, 8]  0.0625 MB         2,304 0.0087890625 MB
     BatchNorm2d-254        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
          Conv2d-255        [1, 256, 8, 8]  0.0625 MB         2,304 0.0087890625 MB
     BatchNorm2d-256        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
          Conv2d-257        [1, 256, 8, 8]  0.0625 MB        65,536         0.25 MB
     BatchNorm2d-258        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
            ReLU-259      [1, 16, 16, 128]   0.125 MB             0          0.0 MB
            ReLU-260      [1, 16, 16, 128]   0.125 MB             0          0.0 MB
          Conv2d-261        [1, 256, 8, 8]  0.0625 MB       131,072          0.5 MB
     BatchNorm2d-262        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
   IdentityLayer-263        [1, 256, 8, 8]  0.0625 MB             0          0.0 MB
          Conv2d-264       [1, 1024, 8, 8]    0.25 MB       263,168   1.00390625 MB
            ReLU-265       [1, 1024, 8, 8]    0.25 MB             0          0.0 MB
          Conv2d-266       [1, 1024, 8, 8]    0.25 MB        10,240    0.0390625 MB
            ReLU-267       [1, 1024, 8, 8]    0.25 MB             0          0.0 MB
          Conv2d-268        [1, 256, 8, 8]  0.0625 MB       262,144          1.0 MB
     BatchNorm2d-269        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
   IdentityLayer-270        [1, 256, 8, 8]  0.0625 MB             0          0.0 MB
          Conv2d-271        [1, 256, 8, 8]  0.0625 MB        65,536         0.25 MB
          Conv2d-272        [1, 256, 8, 8]  0.0625 MB        65,536         0.25 MB
          Conv2d-273        [1, 256, 8, 8]  0.0625 MB        65,536         0.25 MB
          Conv2d-274        [1, 256, 8, 8]  0.0625 MB         2,304 0.0087890625 MB
     BatchNorm2d-275        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
          Conv2d-276        [1, 256, 8, 8]  0.0625 MB         2,304 0.0087890625 MB
     BatchNorm2d-277        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
          Conv2d-278        [1, 256, 8, 8]  0.0625 MB        65,536         0.25 MB
     BatchNorm2d-279        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
          Conv2d-280        [1, 256, 8, 8]  0.0625 MB         2,304 0.0087890625 MB
     BatchNorm2d-281        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
          Conv2d-282        [1, 256, 8, 8]  0.0625 MB         2,304 0.0087890625 MB
     BatchNorm2d-283        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
          Conv2d-284        [1, 256, 8, 8]  0.0625 MB        65,536         0.25 MB
     BatchNorm2d-285        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
          Conv2d-286        [1, 256, 8, 8]  0.0625 MB         2,304 0.0087890625 MB
     BatchNorm2d-287        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
          Conv2d-288        [1, 256, 8, 8]  0.0625 MB         2,304 0.0087890625 MB
     BatchNorm2d-289        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
          Conv2d-290        [1, 256, 8, 8]  0.0625 MB        65,536         0.25 MB
     BatchNorm2d-291        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
            ReLU-292      [1, 16, 16, 128]   0.125 MB             0          0.0 MB
            ReLU-293      [1, 16, 16, 128]   0.125 MB             0          0.0 MB
          Conv2d-294        [1, 256, 8, 8]  0.0625 MB       131,072          0.5 MB
     BatchNorm2d-295        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
   IdentityLayer-296        [1, 256, 8, 8]  0.0625 MB             0          0.0 MB
          Conv2d-297       [1, 1024, 8, 8]    0.25 MB       263,168   1.00390625 MB
            ReLU-298       [1, 1024, 8, 8]    0.25 MB             0          0.0 MB
          Conv2d-299       [1, 1024, 8, 8]    0.25 MB        10,240    0.0390625 MB
            ReLU-300       [1, 1024, 8, 8]    0.25 MB             0          0.0 MB
          Conv2d-301        [1, 256, 8, 8]  0.0625 MB       262,144          1.0 MB
     BatchNorm2d-302        [1, 256, 8, 8]  0.0625 MB           512  0.001953125 MB
   IdentityLayer-303        [1, 256, 8, 8]  0.0625 MB             0          0.0 MB
          Conv2d-304        [1, 768, 8, 8]  0.1875 MB       196,608         0.75 MB
     BatchNorm2d-305        [1, 768, 8, 8]  0.1875 MB         1,536  0.005859375 MB
            ReLU-306        [1, 768, 8, 8]  0.1875 MB             0          0.0 MB
AdaptiveAvgPool2d-307        [1, 768, 1, 1] 0.0029296875 MB             0          0.0 MB
          Linear-308              [1, 512] 0.001953125 MB       393,216          1.5 MB
     BatchNorm1d-309              [1, 512] 0.001953125 MB         1,024   0.00390625 MB
            ReLU-310              [1, 512] 0.001953125 MB             0          0.0 MB
          Linear-311              [1, 100] 0.0003814697265625 MB        51,300 0.1956939697265625 MB
================================================================
Total params: 6,151,172
Trainable params: 6,151,172
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.75
Forward/backward pass size (MB): 216.14
Params size (MB): 23.46
Estimated Total Size (MB): 240.36
----------------------------------------------------------------
[INFO] Register count_convNd() for <class 'torch.nn.modules.conv.Conv2d'>.
[INFO] Register count_normalization() for <class 'torch.nn.modules.batchnorm.BatchNorm2d'>.
[INFO] Register zero_ops() for <class 'torch.nn.modules.activation.ReLU'>.
[INFO] Register zero_ops() for <class 'torch.nn.modules.container.Sequential'>.
[INFO] Register count_adap_avgpool() for <class 'torch.nn.modules.pooling.AdaptiveAvgPool2d'>.
[INFO] Register count_linear() for <class 'torch.nn.modules.linear.Linear'>.
[INFO] Register count_normalization() for <class 'torch.nn.modules.batchnorm.BatchNorm1d'>.
运算量：781.980M, 参数量：6.151M



