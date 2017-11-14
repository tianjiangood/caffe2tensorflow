# -*- coding: utf-8 -:*-
"""
convert a proto file which defiend caffe net define.
"""

import google.protobuf.text_format as tformat
import proto.caffe_pb2 as pb
import tensorflow as tf
from  .util import * 


"""
"""
__tf_op_create_register = {}

def __check_register_exist( key_name ):
    """
    """
    return dict_has_key( __tf_op_create_register,key_name)

def __add_register_creator( key_name,creator ):
    """
    """
    if __check_register_exist( key_name ):
        raise ValueError( '%s is already registered.'%key_name )
    
    __tf_op_create_register[key_name]=creator
    
def __get_register_creator(key_name):
    """
    """
    if( __check_register_exist(key_name) ):
        return __tf_op_create_register[key_name]
    else:
        return None
        
    
def get_register_creator( key_name ):
    """
    """
    return __get_register_creator(key_name)
    
        
def add_register_creator( layer_type,creator ):
    """.
    """
    __add_register_creator(layer_type,creator)
 


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
        

# We can't initialize these variables to 0 - the network will get stuck.
def weight_variable( shape,params ):
    """Create a weight variable with appropriate initialization.
    Parameters
    ----------
    params: pb2.FillerParameter.
    """
    
    if( params.type=='gaussian' ):
        initial = tf.truncated_normal( shape, stddev=params.std,mean=params.mean )
    elif ( params.type=='constant' ):
        initial = tf.constant( params.value, shape=shape)
    else:
        raise ValueError('Unsupport FillerParameter type(%s)'%params.type )
    #elif ( params.type=='uniform' ):
        
    return tf.Variable(initial)

def nn_layer_relu( input_tensors, layer_name, layer_params,add_summary=False ):
    """
    Parameters
    ----------
    input_tensors: .
    layer_params: .
    
    Returns
    -----------
    output_tensors
    """
    if(len(input_tensors) is not 1):
        raise ValueError('input_tensors length(%d) is not 1'%len(input_tensors))
        
    with tf.name_scope(layer_name):
        output_tensor = tf.nn.relu( input_tensors[0] )
        if( add_summary ):
            variable_summaries( output_tensor )
        
    return [output_tensor]
    
def nn_layer_pool( input_tensors, layer_name, layer_params,add_summary=False ):
    """
    Parameters
    ----------
    input_tensors: .
    layer_params: .
    
    Returns
    -----------
    output_tensors
    """
    if(len(input_tensors) is not 1):
        raise ValueError('input_tensors length(%d) is not 1'%len(input_tensors))
        
    kernel_size = [layer_params.kernel_size]*4
    strides = [layer_params.stride]*4  
    output_tensor = None
    
    with tf.name_scope(layer_name):
        if( layer_params.pool==pb.PoolingParameter.MAX ):
            output_tensor = tf.nn.max_pool( input_tensors[0],kernel_size,strides,'SAME')
        elif (layer_params.pool==pb.PoolingParameter.AVE):
            output_tensor = tf.nn.avg_pool( input_tensors[0],kernel_size,strides,'SAME')
        #elif (layer_params.pool==pb.PoolingParameter.STOCHASTIC): 
            #
        else:
            raise ValueError('Unsupport Pooling type(%d)'%layer_params.pool )
            
    return [output_tensor]
   
def nn_layer_conv( input_tensors, layer_name, layer_params,add_summary=False ):
    """
    Parameters
    ----------
    input_tensors: .
    layer_params: .
    
    Returns
    -----------
    output_tensors
    """
    if(len(input_tensors) is not 1):
        raise ValueError('input_tensors length(%d) is not 1'%len(input_tensors))
        
    in_shape = input_tensors[0].shape.as_list()
    output_tensor = None
    #卷积核维度
    filter_shape = [ layer_params.kernel_size[0], 
                     layer_params.kernel_size[0],
                     in_shape[3],
                     layer_params.num_output ]
    if len(layer_params.kernel_size)==2:
        filter_shape[1] = layer_params.kernel_size[1]
        
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable(filter_shape,layer_params.weight_filler )
            if (add_summary):
                variable_summaries(weights)
        
        #strides
        strides = [1,1,1,1]
        if(len(layer_params.stride)>2):
            raise ValueError('layer_params.stride len(%d) is >2'%len(layer_params.stride))
        
        for i in range( 0,len(layer_params.stride)) :
           strides[i+1] = layer_params.stride[i] 
            
        with tf.name_scope('conv'):
            output_tensor = tf.nn.conv2d(input_tensors[0], weights, strides=strides, padding='SAME')
        
        with tf.name_scope('biases'):
            biases = None
            if( layer_params.bias_term ):
                biases = weight_variable( [filter_shape[3]], layer_params.bias_filler )
                if (add_summary):
                    variable_summaries(biases)
                    
        if( biases is not None):    
            output_tensor = tf.nn.bias_add( output_tensor , biases )
        
        if (add_summary):
            tf.summary.histogram('convolution', output_tensor)
        
    return [output_tensor]

#filter_shape = [filter_height,filter_width,in_channel,out_channel]
#input tensor of shape `[batch, in_height, in_width, in_channels]
def nn_layer_fc( input_tensors, layer_name, layer_params,add_summary=False ):
    """
    Parameters
    ----------
    input_tensors: .
    layer_params: .
    
    Returns
    -----------
    output_tensors
    """
    if(len(input_tensors) is not 1):
        raise ValueError('input_tensors length(%d) is not 1'%len(input_tensors))
     
    #flat摊平
    in_shape = input_tensors[0].shape.as_list()
    output_tensor = None
    
    
    #print('in shape',in_shape)
    with tf.name_scope(layer_name):

        flat_dim = 1
        for i in range(1,len(in_shape)):
            flat_dim = flat_dim*in_shape[i]
        
        out_channel = layer_params.num_output 
        with tf.name_scope('weights'):
            weights = weight_variable( [flat_dim,out_channel],layer_params.weight_filler )
            if (add_summary):
                variable_summaries(weights)
                
        with tf.name_scope('biases'):
            biases = None
            if( layer_params.bias_term ):
                biases = weight_variable( [out_channel], layer_params.bias_filler )
                if (add_summary):
                    variable_summaries(biases)
                    
        with tf.name_scope('fc'):
            x_flat = tf.reshape(input_tensors[0], [-1,flat_dim ])
            output_tensor = tf.matmul( x_flat,weights )
            
            if(biases is not None):
                output_tensor = output_tensor + biases
        
    return [output_tensor]
    
    
def caffe2tf( input_net_proto_file,input_shape,phase=None ):
    """caffe2tf.
    
    Parameters
    ----------
    input_net_proto_file : caffe net proto file
    input_shape: [batch, in_height, in_width, in_channels]
        input x dimension.   
    phase : {caffe_pb2.TRAIN, caffe_pb2.TEST, None} optional
        Include layers from this network phase.  If None, include all layers.
        (the default is None)

    Returns
    -------
    tensorflow tensor
    """
    
    #加载网络定义
    netparam = pb.NetParameter()
    
    with open(input_net_proto_file,'r') as fp:
        str_def = fp.read()
        caffe_net = tformat.Parse(str_def,netparam)
    
    #input x
    x = tf.placeholder(tf.float32, input_shape)
    
    #
    layer_tensors = {}
    
    
    #遍历所有的层
    #参考 function: get_pydot_graph@caffe\python\caffe\draw.py
    for layer in caffe_net.layer:
        
        creator = None
        
        if phase is not None:
            included = False
            
            if len(layer.include) == 0:
                included = True
                
            if len(layer.include) > 0 and len(layer.exclude) > 0:
                raise ValueError('layer ' + layer.name + ' has both include '
                                 'and exclude specified.')
                
            for layer_phase in layer.include:
                included = included or layer_phase.phase == phase
            for layer_phase in layer.exclude:
                included = included and not layer_phase.phase == phase
            
            if not included:
                continue
            
        #input layer
        if len(layer.bottom)==0:
            continue
        
        #check0
        creator = get_register_creator(layer.type)
        if( creator is None ):
            raise ValueError( 'layer type %s has no register creator'%layer.type )
            
        #check1
        for bottom_blob in layer.bottom:
            if( not dict_has_key(layer_tensors,bottom_blob.name ) ):
                raise ValueError( 'layer named:%s bottom %s is None,please check flow!!'% \
                                 (layer.name,bottom_blob.name) )
                
        
        
            
        
            
            
        node_label = get_layer_label(layer, rankdir)
        node_name = "%s_%s" % (layer.name, layer.type)
        if (len(layer.bottom) == 1 and len(layer.top) == 1 and
           layer.bottom[0] == layer.top[0]):
            # We have an in-place neuron layer.
            pydot_nodes[node_name] = pydot.Node(node_label,
                                                **NEURON_LAYER_STYLE)
        else:
            layer_style = LAYER_STYLE_DEFAULT
            layer_style['fillcolor'] = choose_color_by_layertype(layer.type)
            pydot_nodes[node_name] = pydot.Node(node_label, **layer_style)
        for bottom_blob in layer.bottom:
            pydot_nodes[bottom_blob + '_blob'] = pydot.Node('%s' % bottom_blob,
                                                            **BLOB_STYLE)
            edge_label = '""'
            pydot_edges.append({'src': bottom_blob + '_blob',
                                'dst': node_name,
                                'label': edge_label})
        for top_blob in layer.top:
            pydot_nodes[top_blob + '_blob'] = pydot.Node('%s' % (top_blob))
            if label_edges:
                edge_label = get_edge_label(layer)
            else:
                edge_label = '""'
            pydot_edges.append({'src': node_name,
                                'dst': top_blob + '_blob',
                                'label': edge_label})
    
    
        
        
    return net_def


    