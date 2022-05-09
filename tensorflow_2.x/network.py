#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 19:09:36 2021

@author: nwoye
"""

import tensorflow as tf

SCOPE           = 'attentiontripnet'
INPUT_SHAPE     = (256,448,3)
OUTPUT_SHAPE    = (256,448,3)
NETSCOPE        = {
            'mobilenet':{
                    'high_level_feature':'out_relu', 
                    'low_level_feature':'block_1_project_BN', 
                    'low_output_shape_ratio':(4,4), 
                    'high_output_shape_ratio':(8,8)
                    },
            'mobilenetv2':{
                    'high_level_feature':'out_relu', 
                    'low_level_feature':'block_1_project_BN', 
                    'low_output_shape_ratio':(4,4), 
                    'high_output_shape_ratio':(8,8)
                    },
            'xception':{
                    'high_level_feature':'block14_sepconv2_act', 
                    'low_level_feature':'block1_conv2_act', 
                    'low_output_shape_ratio':(4,4), 
                    'high_output_shape_ratio':(8,8)
                    },
            'resnet50':{
                    'high_level_feature':'conv5_block3_out', 
                    'low_level_feature':'pool1_pool', 
                    'low_output_shape_ratio':(4,4), 
                    'high_output_shape_ratio':(8,8)
                    },
            'resnet50v2':{
                    'high_level_feature':'post_relu', 
                    'low_level_feature':'pool1_pool', 
                    'low_output_shape_ratio':(4,4), 
                    'high_output_shape_ratio':(8,8)
                    },
            'resnet18v2':{
                    'high_level_feature':'post_relu', 
                    'low_level_feature':'pool1_pool', 
                    'low_output_shape_ratio':(4,4), 
                    'high_output_shape_ratio':(8,8)
                    },
            'densenet169':{
                    'high_level_feature':'bn', 
                    'low_level_feature':'pool1', 
                    'low_output_shape_ratio':(4,4), 
                    'high_output_shape_ratio':(8,8)

                    }
        }    


class AttentionTripnet(tf.keras.Model):
    """
    Rendezvous: surgical action triplet recognition by Nwoye, C.I. et.al. 2020
    @args:
        image_shape: a tuple (height, width) e.g: (224,224)
        basename: Feature extraction network: e.g: "resnet50", "VGG19"
        num_tool: default = 6, 
        num_verb: default = 10, 
        num_target: default = 15, 
        num_triplet: default = 100, 
        dict_map_url: path to the map file for the triplet decomposition
    @call:
        inputs: Batch of input images of shape [batch, height, width, channel]
        training: True or False. Python boolean indicating whether the layer should behave in training mode (adding dropout) or in inference mode (doing nothing)
    @output: 
        enc_i: tuple (cam, logits) for instrument
        enc_v: logits for verb
        enc_t: logits for target
        dec_ivt: logits for triplet
    """
    def __init__(self, image_shape=(256,448,3), basename="resnet50", pretrained='imagenet', num_tool=6, num_verb=10, num_target=15, num_triplet=100, dict_map_url="./"):
        super(AttentionTripnet, self).__init__()
        inputs          = tf.keras.Input(shape=image_shape)
        self.encoder    = Encoder(basename, pretrained, image_shape, num_tool, num_verb, num_target, num_triplet)
        self.decoder    = Decoder(num_tool, num_verb, num_target, num_triplet, dict_map_url)
        enc_i, enc_v, enc_t = self.encoder(inputs)
        dec_ivt         = self.decoder(enc_i, enc_v, enc_t)
        self.attentiontripnet = tf.keras.models.Model(inputs=inputs, outputs=(enc_i, enc_v, enc_t, dec_ivt), name='attentiontripnet')

    def call(self, inputs, training):
        enc_i, enc_v, enc_t, dec_ivt = self.attentiontripnet(inputs, training=training)
        return enc_i, enc_v, enc_t, dec_ivt
    

# Model Encoder
class Encoder(tf.keras.layers.Layer):
    def __init__(self, basename, pretrained, image_shape, num_tool=6, num_verb, num_target, num_triplet):
        super(Encoder, self).__init__()
        self.basemodel  = Basemodel(basename, image_shape, pretrained)
        self.wsl        = WSL(num_tool)
        self.cagam      = CAGAM(num_tool, num_verb, num_target)

    def call(self, inputs, training):
        low_x, high_x   = self.basemodel(inputs, training)
        enc_i           = self.wsl(high_x, training)
        enc_v, enc_t    = self.cagam(high_x, enc_i[0], training)
        return enc_i, enc_v, enc_t



# Backbone
class Basemodel(tf.keras.layers.Layer):
    def __init__(self, basename, image_shape, pretrained='imagenet'):
        super(Basemodel, self).__init__()
        if basename ==  'mobilenet':
            base_model = tf.keras.applications.MobileNetV2(
                                    input_shape=image_shape,
                                    include_top=False,
                                    weights='imagenet')
        elif basename ==  'mobilenetv2':
            base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
                                    input_shape=image_shape,
                                    include_top=False,
                                    weights='imagenet')
        elif basename ==  'xception':
            base_model = tf.keras.applications.Xception(
                                    weights='imagenet',  # Load weights pre-trained on ImageNet.
                                    input_shape=image_shape,
                                    include_top=False)  # Do not include the ImageNet classifier at the top.
        elif basename ==  'resnet50':
            base_model = tf.keras.applications.resnet50.ResNet50(
                                    weights=pretrained,  # Load weights pre-trained on ImageNet.
                                    input_shape=image_shape,
                                    include_top=False)  # Do not include the ImageNet classifier at the top.
        elif basename ==  'resnet50v2':
            base_model = tf.keras.applications.resnet_v2.ResNet50V2(
                                    weights='imagenet',  # Load weights pre-trained on ImageNet.
                                    input_shape=image_shape,
                                    include_top=False)  # Do not include the ImageNet classifier at the top.
        elif basename ==  'resnet18v2':
            import resnet_v2
            base_model = resnet_v2.ResNet18V2(
                                    weights=None,  # Load weights pre-trained on ImageNet.
                                    input_shape=image_shape,
                                    stride=1,
                                    include_top=False)  # Do not include the ImageNet classifier at the top.
        elif basename =='densenet169':
            base_model = tf.keras.applications.densenet.DenseNet169(
                                    include_top=False, 
                                    weights='imagenet',
                                    input_shape=image_shape )
        else: base_model = tf.keras.applications.resnet18.ResNet18( # not impl.
                                    weights='imagenet', # Load weights pre-trained on ImageNet.
                                    input_shape=image_shape,
                                    include_top=False)  # Do not include the ImageNet classifier at the top.
        self.base_model = tf.keras.models.Model(inputs=base_model.input, 
                                                outputs=(base_model.get_layer(NETSCOPE[basename]['low_level_feature']).output, base_model.output),
                                                name='backbone')
        # self.base_model.trainable = trainable        

    def call(self, inputs, training):
        return self.base_model(inputs, training=training)
            
            
# WSL of Tools
class WSL(tf.keras.layers.Layer):
    def __init__(self, num_class, depth=64):
        super(WSL, self).__init__()
        self.num_class = num_class
        self.conv1 = tf.keras.layers.Conv2D(depth, 3, activation=None, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(num_class, 1, activation=None, name='cam')
        self.gmp   = tf.keras.layers.GlobalMaxPooling2D()
        self.bn    = tf.keras.layers.BatchNormalization()
        self.elu   = tf.keras.activations.elu

    def call(self, inputs, training):
        feature = self.conv1(inputs, training=training)
        feature = self.elu(self.bn(feature, training=training))
        cam     = self.conv2(feature, training=training)
        logits  = self.gmp(cam)
        return cam, logits


# Class Activation Guided Attention Mechanism
class CAGAM(tf.keras.layers.Layer):
    def __init__(self, num_tool, num_verb, num_target):
        super(CAGAM, self).__init__()
        depth          = num_tool
        self.context1  = tf.keras.layers.Conv2D(depth, 3, activation=None, padding='same', name='verb/context')
        self.context2  = tf.keras.layers.Conv2D(depth, 3, activation=None, padding='same', name='target/context')
        self.q1        = tf.keras.layers.Conv2D(depth, 1, activation=None, name='verb/query')
        self.q2        = tf.keras.layers.Conv2D(depth, 1, activation=None, name='verb_tool/query')
        self.q3        = tf.keras.layers.Conv2D(depth, 1, activation=None, name='target/query')
        self.q4        = tf.keras.layers.Conv2D(depth, 1, activation=None, name='target_tool/query')
        self.k1        = tf.keras.layers.Conv2D(depth, 1, activation=None, name='verb/key')
        self.k2        = tf.keras.layers.Conv2D(depth, 1, activation=None, name='verb_tool/key')
        self.k3        = tf.keras.layers.Conv2D(depth, 1, activation=None, name='target/key')
        self.k4        = tf.keras.layers.Conv2D(depth, 1, activation=None, name='target_tool/key')
        self.cmap1     = tf.keras.layers.Conv2D(num_verb, 1, activation=None, name='verb/cmap')
        self.cmap2     = tf.keras.layers.Conv2D(num_target, 1, activation=None, name='target/cmap')
        self.gmp       = tf.keras.layers.GlobalMaxPooling2D()
        self.soft      = tf.keras.layers.Softmax()
        self.elu       = tf.keras.activations.elu
        self.beta1     = self.add_weight("encoder/cagam/verb/beta", shape=())
        self.beta2     = self.add_weight("encoder/cagam/target/beta", shape=())       
        self.bn1       = tf.keras.layers.BatchNormalization(name='verb/batch_normalization_1')
        self.bn2       = tf.keras.layers.BatchNormalization(name='verb/batch_normalization_2')
        self.bn3       = tf.keras.layers.BatchNormalization(name='verb/batch_normalization_3')
        self.bn4       = tf.keras.layers.BatchNormalization(name='verb/batch_normalization_4')
        self.bn5       = tf.keras.layers.BatchNormalization(name='verb/batch_normalization_5')
        self.bn6       = tf.keras.layers.BatchNormalization(name='verb/batch_normalization_6')
        self.bn7       = tf.keras.layers.BatchNormalization(name='target/batch_normalization_1')
        self.bn8       = tf.keras.layers.BatchNormalization(name='target/batch_normalization_2')
        self.bn9       = tf.keras.layers.BatchNormalization(name='target/batch_normalization_3')
        self.bn10      = tf.keras.layers.BatchNormalization(name='target/batch_normalization_4')
        self.bn11      = tf.keras.layers.BatchNormalization(name='target/batch_normalization_5')
        self.bn12      = tf.keras.layers.BatchNormalization(name='target/batch_normalization_6')

    def get_verb(self, raw, cam, training):
        x = self.elu(self.bn1(self.context1(raw, training=training), training=training))
        z = tf.identity(x)               
        q1 = self.elu(self.bn2(self.q1(x, training=training), training=training))
        k1 = self.elu(self.bn3(self.k1(x, training=training), training=training))
        s  = k1.get_shape().as_list()
        w1 = tf.matmul(tf.reshape(q1,[-1,s[1]*s[2],s[3]]), tf.reshape(k1,[-1,s[1]*s[2],s[3]]), transpose_a=True)        
        q2 = self.elu(self.bn4(self.q2(cam, training=training), training=training))
        k2 = self.elu(self.bn5(self.k2(cam, training=training), training=training))
        s  = k2.get_shape().as_list()
        dk = tf.cast(s[-1], tf.float32)
        w2 = tf.matmul(tf.reshape(q2,[-1,s[1]*s[2],s[3]]), tf.reshape(k2,[-1,s[1]*s[2],s[3]]), transpose_a=True)
        attention = (w1 * w2) / tf.sqrt(dk)  
        attention = self.soft(attention) 
        s = z.get_shape().as_list()        
        v = tf.reshape(z, [-1, s[1]*s[2], s[3]])
        e = tf.matmul(v, attention) * self.beta1
        e = tf.reshape(e, [-1, s[1], s[2], s[3]]) 
        e = self.bn6(e + z)
        cmap = self.cmap1(e, training=training)
        y = self.gmp(cmap)
        return cmap, y

    def get_target(self, raw, cam, training):    
        x = self.elu(self.bn7(self.context2(raw, training=training), training=training))
        z = tf.identity(x)       
        q3 = self.elu(self.bn8(self.q3(x, training=training), training=training))
        k3 = self.elu(self.bn9(self.k3(x, training=training), training=training))
        s  = k3.get_shape().as_list()
        w3 = tf.matmul(tf.reshape(q3,[-1,s[1]*s[2],s[3]]), tf.reshape(k3,[-1,s[1]*s[2],s[3]]), transpose_b=True)  
        q4 = self.elu(self.bn10(self.q4(cam, training=training), training=training))
        k4 = self.elu(self.bn11(self.k4(cam, training=training), training=training))
        s  = k4.get_shape().as_list()
        dk = tf.cast(s[-1], tf.float32)
        w4 = tf.matmul(tf.reshape(q4,[-1,s[1]*s[2],s[3]]), tf.reshape(k4,[-1,s[1]*s[2],s[3]]), transpose_b=True)
        attention = (w3 * w4) / tf.sqrt(dk)  
        attention = self.soft(attention)
        s = z.get_shape().as_list() 
        v = tf.reshape(z, [-1, s[1]*s[2], s[3]])
        e = tf.matmul(attention, v) * self.beta2
        e = tf.reshape(e, [-1, s[1], s[2], s[3]])
        e = self.bn12(e + z)
        cmap = self.cmap2(e, training=training)
        y = self.gmp(cmap)
        return cmap, y

    def call(self, inputs, cam, training):
        cam_v, logit_v = self.get_verb(inputs, cam, training)
        cam_t, logit_t = self.get_target(inputs, cam, training)
        return (cam_v, logit_v), (cam_t, logit_t)


# 3D interaction space
class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_tool, num_verb, num_target, num_triplet, dict_map_url):
        super(Decoder, self).__init__()
        self.num_tool       = num_tool
        self.num_verb       = num_verb
        self.num_target     = num_target
        self.valid_position = self.constraint(num_verb, num_target, url=os.path.join(dict_map_url, 'maps.txt'))
        self.alpha          = self.add_weight("decoder/3dis/triplet/alpha", shape=(self.num_tool, self.num_tool))
        self.beta           = self.add_weight("decoder/3dis/triplet/beta", shape=(self.num_verb, self.num_verb))
        self.gamma          = self.add_weight("decoder/3dis/triplet/gamma", shape=(self.num_target, self.num_target))
        self.fc             = tf.keras.layers.Dense(num_triplet, name='mlp')

    def constraint(self, num_verb, num_target, url):
        # 3D Interaction space constraints mask
        indexes = []
        with open(url) as f:              
            for line in f:
                values = line.split(',')
                if '#' in values[0]:
                    continue
                indexes.append( list(map(int, values[1:4])) )
            indexes = np.array(indexes)
        index_pos = []  
        for index in indexes:
            index_pos.append(index[0]*(num_target*num_verb) + index[1]*(num_target) + index[2])            
        return np.array(index_pos)

    def mask(self, ivts):
        # Map 3D ispace to a vector of valid triplets
        ivt_flatten    = tf.reshape(ivts, [tf.shape(ivts)[0], self.num_tool*self.num_verb*self.num_target])
        valid_triplets = tf.gather(params=ivt_flatten, indices=self.valid_position, axis=-1, name="extract_valid_triplet")
        return valid_triplets

    def call(self, tool_logits, verb_logits, target_logits, is_training):
        tool      = tf.matmul(tool_logits, self.alpha, name='alpha_tool')
        verb      = tf.matmul(verb_logits, self.beta, name='beta_verb')
        target    = tf.matmul(target_logits, self.gamma, name='gamma_target')   
        ivt_maps  = tf.einsum('bi,bv,bt->bivt', tool, verb, target ) 
        ivt_mask  = self.mask(ivts=ivt_maps)  
        ivt_mask  = self.fc(self.fc)
        return ivt_mask  
