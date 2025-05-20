
cnn_encoder_config = {
    "branch_cnn_1": { "num_layers": 5,
                     "input_size_list": [1, 32, 64, 128, 256],
                     "hidden_size_list": [32, 64, 128, 256, 16],
                     "conv_kernel_size_list": [258, 2, 2, 2, 16],
                     "conv_stride_list": [2, 2, 2, 2, 2] },
    
    "branch_cnn_2": { "num_layers": 4, 
                     "input_size_list": [1, 32, 64, 128], 
                     "hidden_size_list": [32, 64, 128, 128], 
                     "conv_kernel_size_list": [3, 3, 3, 3], 
                     "conv_stride_list": [2, 2, 2, 2],
                     "pool_kernel_size_list": [2, 2, 2, 2],
                     "pool_stride_list": [2, 2, 2, 2],
                     },
    
    "branch_cnn_3": { "num_layers": 5,
                     "input_size_list": [1, 32, 64, 128, 256],
                     "hidden_size_list": [32, 64, 128, 256, 128],
                     "conv_kernel_size_list": [258, 2, 2, 2, 16],
                     "conv_stride_list": [2, 2, 2, 2, 2],
                     "padding_list": [0, 0, 0, 0, 0]
                     },
    
    "branch_cnn_4": { "num_layers": 9,
                     "input_size_list": [1, 16, 32, 64, 128, 256, 512, 512, 512],
                     "hidden_size_list": [16, 32, 64, 128, 256, 512, 512, 512, 512],
                     "conv_kernel_size_list": [5,5,5,5,5,5,5,3,3],
                     "conv_stride_list": [2, 2, 2, 2, 2, 2, 2, 2, 2],
                     "padding_list": [2,2,2,2,2,2,2,1,1]
                     },
    
    "branch_cnn_5": { "num_layers": 5,
                     "input_size_list": [1, 16, 32, 64, 128],
                     "hidden_size_list": [16, 32, 64, 128, 400],
                     "conv_kernel_size_list": [5, 5, 3, 3, 3],
                     "conv_stride_list": [3, 3, 2, 2, 2],
                     "padding_list": [0, 0, 0, 0, 0]
                     },
    
    "case1_cnn_1": { "num_layers": 3,
                     "input_size_list": [1, 64, 128],
                     "hidden_size_list": [64, 128, 512],
                     "conv_kernel_size_list": [3, 3, 3],
                     "conv_stride_list": [2, 2, 2],
                     "padding_list": [0, 0, 0]
                     },
    
    "case1_cnn_2": { "num_layers": 3,
                     "input_size_list": [1, 64, 128],
                     "hidden_size_list": [64, 128, 256],
                     "conv_kernel_size_list": [2, 2, 2],
                     "conv_stride_list": [1, 1, 1],
                     "padding_list": [0, 0, 0],
                     },
    
    "branch_cnn_6": { "num_layers": 6,
                     "input_size_list": [1, 16, 32, 64, 128, 256, 512],
                     "hidden_size_list": [16, 32, 64, 128, 256, 512],
                     "conv_kernel_size_list": [3, 3, 3, 3, 3,3],
                     "conv_stride_list": [3, 2, 2, 2 ,2,2],
                     "padding_list": [0, 0, 0, 0, 0,0]
                     },
    
    "branch_cnn_7": { "num_layers": 6,
                     "input_size_list": [1, 16, 32, 64, 128, 256, 512],
                     "hidden_size_list": [16, 32, 64, 128, 256, 512],
                     "conv_kernel_size_list": [3, 3, 3, 3, 3, 4],
                     "conv_stride_list": [2, 2, 2, 2, 3, 1],
                     "padding_list": [1, 1, 1, 0, 1, 0]
                     },
    "branch_cnn_8": { "num_layers": 6,
                     "input_size_list": [1, 16, 32, 64, 128, 256, 512],
                     "hidden_size_list": [16, 32, 64, 128, 256, 512],
                     "conv_kernel_size_list": [3, 3, 3, 3, 3, 4],
                     "conv_stride_list": [3, 3, 2, 2, 3, 2],
                     "padding_list": [1, 1, 1, 0, 1, 0]
                     },
    
    "aeonet_cnn_1": { "num_layers": 6,
                     "input_size_list": [1, 16, 32, 64, 128, 256, 256],
                     "hidden_size_list": [16, 32, 64, 128, 256, 256],
                     "conv_kernel_size_list": [3, 3, 3, 3, 3, 4],
                     "conv_stride_list": [2, 2, 2, 2 , 3, 1],
                     "padding_list": [1, 1, 1, 0, 1, 0]
                     },
    
    "aeonet_cnn_2": { "num_layers": 4,
                     "input_size_list": [1, 100, 200, 400, 200, 100],
                     "hidden_size_list": [100, 200, 400, 200, 100, 4],
                     "conv_kernel_size_list": [3, 3, 3, 3, 3, 3],
                     "conv_stride_list": [2, 2, 2, 2, 2, 2],
                     "padding_list": [0, 0, 0, 0, 0, 0]
                     },
    
    "aeonet_cnn_3": { "num_layers": 4,
                     "input_size_list": [1, 2, 2, 4],
                     "hidden_size_list": [2, 2, 4, 4],
                     "conv_kernel_size_list": [3, 3, 3, 3],
                     "conv_stride_list": [1, 1, 1, 1],
                     "padding_list": [1, 1, 1, 1]
                     },
    
    
    "trunk_cnn_1": { "num_layers": 5,
                     "input_size_list": [4, 32, 64, 128, 256],
                     "hidden_size_list": [32, 64, 128, 256, 16],
                     "conv_kernel_size_list": [258, 2, 2, 2, 16],
                     "conv_stride_list": [2, 2, 2, 2, 2] },
    
    "trunk_cnn_2": { "num_layers": 4, 
                     "input_size_list": [4, 32, 64, 128], 
                     "hidden_size_list": [32, 64, 128, 128], 
                     "conv_kernel_size_list": [3, 3, 3, 3], 
                     "conv_stride_list": [2, 2, 2, 2],
                     "pool_kernel_size_list": [2, 2, 2, 2],
                     "pool_stride_list": [2, 2, 2, 2],
                     },   
    "decoder_cnn_1": { "num_layers": 4,
                     "input_size_list": [4, 16, 64, 128],
                     "hidden_size_list": [16, 64, 128, 4],
                     "conv_kernel_size_list": [5, 5, 5, 5],
                     "conv_stride_list": [1, 1, 1, 1],
                     "padding_list": ['same', 'same', 'same', 'same']
                     },
    "decoder_cnn_case1": { "num_layers": 4,
                     "input_size_list": [1, 16, 64, 64],
                     "hidden_size_list": [16, 64, 64, 1],
                     "conv_kernel_size_list": [5, 5, 5, 5],
                     "conv_stride_list": [1, 1, 1, 1],
                     "padding_list": ['same', 'same', 'same', 'same']
                     },
    "params_cnn_1": { "num_layers": 4,
                     "input_size_list": [1, 16, 64, 128],
                     "hidden_size_list": [16, 64, 128, 1],
                     "conv_kernel_size_list": [5, 5, 5, 5],
                     "conv_stride_list": [1, 1, 1, 1],
                     "padding_list": ['same', 'same', 'same', 'same']
                     },
    "params_refine_cnn_1": { "num_layers": 4,
                     "input_size_list": [1, 16, 64, 128],
                     "hidden_size_list": [16, 64, 128, 1],
                     "conv_kernel_size_list": [3, 5, 5, 5],
                     "conv_stride_list": [2, 1, 1, 1],
                     "padding_list": [1, 'same', 'same', 'same'],
                     },
    "decoder_cnn_case1_2": { "num_layers": 4,
                     "input_size_list": [1, 16, 64, 64],
                     "hidden_size_list": [16, 64, 64, 1],
                     "conv_kernel_size_list": [3, 3, 3, 3],
                     "conv_stride_list": [1, 1, 1, 1],
                     "padding_list": ['same', 'same', 'same', 'same']
                     },
    "decoder_cnn_2": { "num_layers": 3,
                     "input_size_list": [4, 16, 64],
                     "hidden_size_list": [16, 64, 4],
                     "conv_kernel_size_list": [5, 5, 5],
                     "conv_stride_list": [1, 1, 1],
                     "padding_list": ['same', 'same', 'same']
                     },
    "grid_1": { "num_layers": 4,
                     "input_size_list": [1, 64, 128, 256],
                     "hidden_size_list": [64, 128, 256, 1],
                     "conv_kernel_size_list": [3, 3, 3, 3],
                     "conv_stride_list": [1, 1, 1, 1],
                     "padding_list": ['same', 'same', 'same', 'same'],
                     }
}

cnn_decoder_config = {
    
    
    'aeonet_decnn_1': { "num_layers": 6,
                        "input_size_list": [256, 256, 128, 64, 32, 16],
                        "hidden_size_list": [256, 128, 64, 32, 16, 4],
                        "conv_kernel_size_list": [4, 3, 3, 3, 3,3],
                        "conv_stride_list": [1, 3, 2, 2 , 2, 2],
                        "padding_list": [0, 0, 0, 1, 1, 1],
                        'output_padding_list': [0, 0, 0, 1, 1, 1],
                        },
    
    "trunk_decnn_1": { "num_layers": 5,
                      "input_size_list": [16, 256, 128, 64, 32],
                     "hidden_size_list": [256, 128, 64, 32, 4],
                     "conv_kernel_size_list": [16, 2, 2, 2, 258],
                     "conv_stride_list": [2, 2, 2, 2, 2] },
    
    "trunk_decnn_2": { "num_layers": 5,
                      "input_size_list": [512, 256, 128, 64, 32],
                     "hidden_size_list": [256, 128, 64, 32, 4],
                     "conv_kernel_size_list": [16, 2, 2, 2, 258],
                     "conv_stride_list": [2, 2, 2, 2, 2],
                     "padding_list": [0, 0, 0, 0, 0],
                     "output_padding_list": [0, 0, 0, 0, 0]
                     },
    
    "trunk_decnn_4": { "num_layers": 9,
                      "input_size_list": [512, 512, 512, 512, 512, 256, 128, 64, 32],
                      "hidden_size_list": [512, 512, 512, 512, 256, 128, 64, 32, 4],
                      "conv_kernel_size_list": [3, 3, 5, 5, 5, 5, 5, 5, 5],
                      "conv_stride_list": [2, 2, 2, 2, 2, 2, 2, 2, 2],
                      "padding_list": [1,1,2,2,2,2,2,2,2],
                      "output_padding_list": [1,1,1,1,1,1,1,1,1],
                      },
    
    "trunk_decnn_5": { "num_layers": 5,
                     "input_size_list": [200, 128, 64, 32, 16],
                     "hidden_size_list": [128, 64, 32, 16, 3],
                     "conv_kernel_size_list": [3, 3, 3, 5, 5],
                     "conv_stride_list": [2, 2, 2, 3, 3],
                     "padding_list": [0, 0, 0, 0, 0],
                     "output_padding_list": [1, 1, 0, 1, 0],
                     },
    
    "case1_decnn_1": { "num_layers": 3,
                     "input_size_list": [512, 128, 64],
                     "hidden_size_list": [128, 64, 1],
                     "conv_kernel_size_list": [3, 3, 3],
                     "conv_stride_list": [2, 2, 2],
                     "padding_list": [0, 0, 0],
                     "output_padding_list": [0, 0, 1],
                     },
    
    "case1_decnn_2": { "num_layers": 3,
                     "input_size_list": [256, 128, 64],
                     "hidden_size_list": [128, 64, 1],
                     "conv_kernel_size_list": [2, 2, 2],
                     "conv_stride_list": [1, 1, 1],
                     "padding_list": [0, 0, 0],
                     "output_padding_list": [1, 0, 0],
                     },
    
    
    "trunk_decnn_5_1": { "num_layers": 5,
                     "input_size_list": [400, 128, 64, 32, 16],
                     "hidden_size_list": [128, 64, 32, 16, 1],
                     "conv_kernel_size_list": [3, 3, 3, 5, 5],
                     "conv_stride_list": [2, 2, 2, 3, 3],
                     "padding_list": [0, 0, 0, 0, 0],
                     "output_padding_list": [1, 1, 0, 1, 0],
                     },
    
    "trunk_decnn_6": { "num_layers": 6,
            "input_size_list": [512, 256, 128, 64, 32, 16, 1],
            "hidden_size_list": [256, 128, 64, 32, 16, 4],
            "conv_kernel_size_list": [3, 3, 3, 3, 3,3],
            "conv_stride_list": [2, 2, 2, 2 , 2,3],
            "padding_list": [0, 0, 0, 0, 0,0],
            'output_padding_list': [0, 0, 0, 1, 1,2],
            },
    
    "trunk_decnn_7": { "num_layers": 6,
                "input_size_list": [512, 256, 128, 64, 32, 16],
                "hidden_size_list": [256, 128, 64, 32, 16, 4],
                "conv_kernel_size_list": [4, 3, 3, 3, 3,3],
                "conv_stride_list": [1, 3, 2, 2 , 2, 2],
                "padding_list": [0, 0, 0, 1, 1, 1],
                'output_padding_list': [0, 0, 0, 1, 1, 1],
                },
    
    "trunk_decnn_8": { "num_layers": 6,
                "input_size_list": [200, 256, 128, 64, 32, 16, 1],
                "hidden_size_list": [256, 128, 64, 32, 16, 4],
                "conv_kernel_size_list": [4, 3, 3, 3, 3,3],
                "conv_stride_list": [1, 3, 2, 2 , 2, 2],
                "padding_list": [0, 0, 0, 1, 1, 1],
                'output_padding_list': [0, 0, 0, 1, 1, 1],
                }
    
    }


cnn_config = {
    
    "net_1": { "num_layers": 6,
                     "input_size_list": [1, 64, 128, 512, 256, 128],
                     "hidden_size_list": [64, 128, 512, 256, 128, 1],
                     "conv_kernel_size_list": [2, 2, 2, 2, 2, 2],
                     "conv_stride_list": [1, 1, 1, 1, 1, 1],
                     "padding_list": [0, 0, 1, 1, 0, 0],
                     },
     "net_2":{ "num_layers": 9,
                     "input_size_list": [1, 64, 128, 256, 400, 512, 256, 128, 64],
                     "hidden_size_list": [64, 128, 256, 400, 512 ,256, 128, 64, 4],
                     "conv_kernel_size_list": [3, 3, 3, 3, 3, 3,3,3,3],
                     "conv_stride_list": [1, 1, 1, 1, 1, 1,1,1,1],
                     "padding_list": [1, 1, 1, 1, 1, 1,0,0,0],
                     }
    
}