# -*- coding: utf-8 -*-

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import caffe2tf.proto2tf as cf2tf 

def parse_args():
    """Parse input arguments
    """

    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--input_net_proto_file',
                        type=str,
                        help='Input network prototxt file',
                        default='./net.deploy' )
    
    parser.add_argument('--output_image_file',
                        type=str,
                        help='Output image file',
                        default='./net.deploy' )
    
    parser.add_argument('--debug',
                        type=bool,
                        help=('debug switch True/False'),
                        default=False )
    
    args = parser.parse_args()
    
    return args


def main():
    
    args = parse_args()
    
    tf_graph = cf2tf.caffe2tf(args.input_net_proto_file)
    
    
if __name__=='__main__':
    main()