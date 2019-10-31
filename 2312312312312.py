from tensorflow.python.keras.utils.data_utils import get_file


weights_path = get_file('deeplabv3_xception_tf_dim_ordering_tf_kernels_cityscapes.h5',
                                    'sdfsdf',
                                    cache_dir='models',
                        cache_subdir='.')


print(weights_path)


