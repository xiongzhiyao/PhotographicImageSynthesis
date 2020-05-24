from __future__ import division
import os,helper,time,scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

import os
import helper
import time
import scipy.io
import scipy
import json
import PIL
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.transform import resize

shape = (256, 512)

occlusion = np.arange(19) #number2 is building

mask_classes = [
    100,  # glass
    200,  # Brick
    202,  # Stone
    203,  # Metal
    209,  # Stucco
    211,  # Concrete
    220,  # Tudor
    221,  # Wrap
    222,  # Unknown
    400,  # Shutter
    401,  # Trim
    402,  # Fascia
    403,  # Soffit
    404,  # Roof
    405,  # Sash
    407,  # Vent
    408,  # Door Package
    409,  # Opening's Trim
    601,  # Horizontal Siding
    602,  # Vertical Siding
    603,  # Shingled SIding
    604,  # Other Siding
]

palette = []
palette.append((255,0,0))
def init_color_mapping():
    max_color_hex = 256 * 256 * 256 - 1
    color_step = np.floor(max_color_hex / len(mask_classes))
    for ix in range(len(mask_classes)):
        key_color_num = (ix + 1) * color_step
        key_color_num = int(key_color_num)
        key_color_num = format(key_color_num, 'x')
        while len(key_color_num) < 6:
            key_color_num = '0' + key_color_num
        rgb = tuple(int(key_color_num[i:i+2], 16) for i in (0, 2, 4))
        palette.append(rgb)

init_color_mapping()

def get_semantic_map(mask_path, occlusion_path):
    mask = Image.open(mask_path).convert('RGB')
    mask = mask.resize((shape[1], shape[0]), resample = PIL.Image.NEAREST)
    mask = np.array(mask)

    tmp=np.zeros((shape[0],shape[1]),dtype=np.float32)
    for k in range(len(palette)):
        matched = (mask[:,:,0]==palette[k][0])&(mask[:,:,1]==palette[k][1])&(mask[:,:,2]==palette[k][2])
        tmp[matched] = k

    occlusion = scipy.sparse.load_npz(occlusion_path)
    occlusion = occlusion.todense()
    occlusion = resize(occlusion, shape, preserve_range=True)

    idx = len(palette)
    for i in range(19):
        if i == 2:
            continue
        tmp[occlusion == i] = idx
        idx += 1

    output = np.zeros((shape[0],shape[1],len(palette) + 19 - 1),dtype=np.float32)
    for i in range(output.shape[2]):
        output[:,:,i] = (tmp == i)

    return output.reshape((1,)+output.shape)


data, raw_data = [], []
with open('data/snowman_crn_train_20200521.man', 'r') as fd:
    raw_data = fd.readlines()
for d in raw_data:
    pair = json.loads(d)
    data.append(pair)

# for d in data:
#     image_id = d['camera']
#     image_path = os.path.join('/home/ec2-user/CRN/hover_data/image', '{}.jpg'.format(image_id))
#     mask_path = os.path.join('/home/ec2-user/CRN/hover_data/mask', '{}.jpg'.format(image_id))
#     occlusion_path = os.path.join('/home/ec2-user/CRN/hover_data/occlusion', '{}.npz'.format(image_id))
#     img = np.array(Image.open(image_path).resize((shape[1], shape[0]), resample = PIL.Image.NEAREST))
#     get_semantic_map(mask_path, occlusion_path)
#     np.expand_dims(np.float32(img),axis=0)#training image
#     print('here')


#========================================================================================
def lrelu(x):
    return tf.maximum(0.2*x,x)

def build_net(ntype,nin,nwb=None,name=None):
    if ntype=='conv':
        return tf.nn.relu(tf.nn.conv2d(nin,nwb[0],strides=[1,1,1,1],padding='SAME',name=name)+nwb[1])
    elif ntype=='pool':
        return tf.nn.avg_pool(nin,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def get_weight_bias(vgg_layers,i):
    weights=vgg_layers[i][0][0][2][0][0]
    weights=tf.constant(weights)
    bias=vgg_layers[i][0][0][2][0][1]
    bias=tf.constant(np.reshape(bias,(bias.size)))
    return weights,bias

def build_vgg19(input,reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    net={}
    vgg_rawnet=scipy.io.loadmat('VGG_Model/imagenet-vgg-verydeep-19.mat')
    vgg_layers=vgg_rawnet['layers'][0]
    net['input']=input-np.array([123.6800, 116.7790, 103.9390]).reshape((1,1,1,3))
    net['conv1_1']=build_net('conv',net['input'],get_weight_bias(vgg_layers,0),name='vgg_conv1_1')
    net['conv1_2']=build_net('conv',net['conv1_1'],get_weight_bias(vgg_layers,2),name='vgg_conv1_2')
    net['pool1']=build_net('pool',net['conv1_2'])
    net['conv2_1']=build_net('conv',net['pool1'],get_weight_bias(vgg_layers,5),name='vgg_conv2_1')
    net['conv2_2']=build_net('conv',net['conv2_1'],get_weight_bias(vgg_layers,7),name='vgg_conv2_2')
    net['pool2']=build_net('pool',net['conv2_2'])
    net['conv3_1']=build_net('conv',net['pool2'],get_weight_bias(vgg_layers,10),name='vgg_conv3_1')
    net['conv3_2']=build_net('conv',net['conv3_1'],get_weight_bias(vgg_layers,12),name='vgg_conv3_2')
    net['conv3_3']=build_net('conv',net['conv3_2'],get_weight_bias(vgg_layers,14),name='vgg_conv3_3')
    net['conv3_4']=build_net('conv',net['conv3_3'],get_weight_bias(vgg_layers,16),name='vgg_conv3_4')
    net['pool3']=build_net('pool',net['conv3_4'])
    net['conv4_1']=build_net('conv',net['pool3'],get_weight_bias(vgg_layers,19),name='vgg_conv4_1')
    net['conv4_2']=build_net('conv',net['conv4_1'],get_weight_bias(vgg_layers,21),name='vgg_conv4_2')
    net['conv4_3']=build_net('conv',net['conv4_2'],get_weight_bias(vgg_layers,23),name='vgg_conv4_3')
    net['conv4_4']=build_net('conv',net['conv4_3'],get_weight_bias(vgg_layers,25),name='vgg_conv4_4')
    net['pool4']=build_net('pool',net['conv4_4'])
    net['conv5_1']=build_net('conv',net['pool4'],get_weight_bias(vgg_layers,28),name='vgg_conv5_1')
    net['conv5_2']=build_net('conv',net['conv5_1'],get_weight_bias(vgg_layers,30),name='vgg_conv5_2')
    net['conv5_3']=build_net('conv',net['conv5_2'],get_weight_bias(vgg_layers,32),name='vgg_conv5_3')
    net['conv5_4']=build_net('conv',net['conv5_3'],get_weight_bias(vgg_layers,34),name='vgg_conv5_4')
    net['pool5']=build_net('pool',net['conv5_4'])
    return net

def recursive_generator(label,sp):
    dim=512 if sp>=128 else 1024
    if sp==4:
        input=label
    else:
        downsampled=tf.image.resize_area(label,(sp//2,sp),align_corners=False)
        input=tf.concat([tf.image.resize_bilinear(recursive_generator(downsampled,sp//2),(sp,sp*2),align_corners=True),label],3)
    net=slim.conv2d(input,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='g_'+str(sp)+'_conv1')
    net=slim.conv2d(net,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='g_'+str(sp)+'_conv2')
    if sp==256:
        net=slim.conv2d(net,27,[1,1],rate=1,activation_fn=None,scope='g_'+str(sp)+'_conv100')
        net=(net+1.0)/2.0*255.0
        split0,split1,split2=tf.split(tf.transpose(net,perm=[3,1,2,0]),num_or_size_splits=3,axis=0)
        net=tf.concat([split0,split1,split2],3)
    return net

def compute_error(real,fake,label):
    return tf.reduce_mean(label*tf.expand_dims(tf.reduce_mean(tf.abs(fake-real),reduction_indices=[3]),-1),reduction_indices=[1,2])#diversity loss

#os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
#os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax([int(x.split()[2]) for x in open('tmp','r').readlines()]))#select a GPU with maximum available memory
#os.system('rm tmp')
sess=tf.Session()
is_training=True
sp=256#spatial resolution: 256x512
with tf.variable_scope(tf.get_variable_scope()):
    label=tf.placeholder(tf.float32,[None,None,None,42])
    real_image=tf.placeholder(tf.float32,[None,None,None,3])
    fake_image=tf.placeholder(tf.float32,[None,None,None,3])
    generator=recursive_generator(label,sp)
    weight=tf.placeholder(tf.float32)
    vgg_real=build_vgg19(real_image)
    vgg_fake=build_vgg19(generator,reuse=True)
    p0=compute_error(vgg_real['input'],vgg_fake['input'],label)
    p1=compute_error(vgg_real['conv1_2'],vgg_fake['conv1_2'],label)/1.6
    p2=compute_error(vgg_real['conv2_2'],vgg_fake['conv2_2'],tf.image.resize_area(label,(sp//2,sp)))/2.3
    p3=compute_error(vgg_real['conv3_2'],vgg_fake['conv3_2'],tf.image.resize_area(label,(sp//4,sp//2)))/1.8
    p4=compute_error(vgg_real['conv4_2'],vgg_fake['conv4_2'],tf.image.resize_area(label,(sp//8,sp//4)))/2.8
    p5=compute_error(vgg_real['conv5_2'],vgg_fake['conv5_2'],tf.image.resize_area(label,(sp//16,sp//8)))*10/0.8#weights lambda are collected at 100th epoch
    content_loss=p0+p1+p2+p3+p4+p5
    G_loss=tf.reduce_sum(tf.reduce_min(content_loss,reduction_indices=0))*0.999+tf.reduce_sum(tf.reduce_mean(content_loss,reduction_indices=0))*0.001
lr=tf.placeholder(tf.float32)
G_opt=tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss,var_list=[var for var in tf.trainable_variables() if var.name.startswith('g_')])
saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())
# ckpt=tf.train.get_checkpoint_state("result_256p")
# if ckpt:
#     print('loaded '+ckpt.model_checkpoint_path)
#     saver.restore(sess,ckpt.model_checkpoint_path)

if is_training:
    data_len = len(data)
    g_loss=np.zeros(data_len,dtype=float)
    input_images=[None]*data_len
    label_images=[None]*data_len
    for epoch in range(1,201):
        if os.path.isdir("result_256p/%04d"%epoch):
            continue
        cnt=0
        for ind in np.random.permutation(data_len) - 10:
            st=time.time()
            cnt+=1

            try:
                if input_images[ind] is None:
                    
                    d = data[ind]
                    image_id = d['camera']
                    image_path = os.path.join('/home/ec2-user/CRN/hover_data/image', '{}.jpg'.format(image_id))
                    mask_path = os.path.join('/home/ec2-user/CRN/hover_data/mask', '{}.jpg'.format(image_id))
                    occlusion_path = os.path.join('/home/ec2-user/CRN/hover_data/occlusion', '{}.npz'.format(image_id))
                    img = np.array(Image.open(image_path).resize((shape[1], shape[0]), resample = PIL.Image.NEAREST))

                    label_images[ind] = get_semantic_map(mask_path, occlusion_path)
                    input_images[ind]=np.expand_dims(np.float32(img),axis=0)#training image

                _,G_current,l0,l1,l2,l3,l4,l5=sess.run([G_opt,G_loss,p0,p1,p2,p3,p4,p5],feed_dict={label:np.concatenate((label_images[ind],np.expand_dims(1-np.sum(label_images[ind],axis=3),axis=3)),axis=3),real_image:input_images[ind],lr:1e-4})#may try lr:min(1e-6*np.power(1.1,epoch-1),1e-4 if epoch>100 else 1e-3) in case lr:1e-4 is not good
                g_loss[ind]=G_current
                print("%d %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f"%(epoch,cnt,np.mean(g_loss[np.where(g_loss)]),np.mean(l0),np.mean(l1),np.mean(l2),np.mean(l3),np.mean(l4),np.mean(l5),time.time()-st))
            except Exception as e:
                print(e)

        os.makedirs("result_256p/%04d"%epoch)
        target=open("result_256p/%04d/score.txt"%epoch,'w')
        target.write("%f"%np.mean(g_loss[np.where(g_loss)]))
        target.close()
        saver.save(sess,"result_256p/model.ckpt")
        saver.save(sess,"result_256p/%04d/model.ckpt"%epoch)
        for ind in range(5000,5010):
            d = data[ind]
            image_id = d['camera']
            image_path = os.path.join('/home/ec2-user/CRN/hover_data/image', '{}.jpg'.format(image_id))
            mask_path = os.path.join('/home/ec2-user/CRN/hover_data/mask', '{}.jpg'.format(image_id))
            occlusion_path = os.path.join('/home/ec2-user/CRN/hover_data/occlusion', '{}.npz'.format(image_id))
            semantic = get_semantic_map(mask_path, occlusion_path)
            output=sess.run(generator,feed_dict={label:np.concatenate((semantic,np.expand_dims(1-np.sum(semantic,axis=3),axis=3)),axis=3)})
            output=np.minimum(np.maximum(output,0.0),255.0)
            upper=np.concatenate((output[0,:,:,:],output[1,:,:,:],output[2,:,:,:]),axis=1)
            middle=np.concatenate((output[3,:,:,:],output[4,:,:,:],output[5,:,:,:]),axis=1)
            bottom=np.concatenate((output[6,:,:,:],output[7,:,:,:],output[8,:,:,:]),axis=1)
            scipy.misc.toimage(np.concatenate((upper,middle,bottom),axis=0),cmin=0,cmax=255).save("result_256p/%04d/%06d_output.jpg"%(epoch,ind))
