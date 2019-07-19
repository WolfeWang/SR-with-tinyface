#coding: utf-8
import tensorflow as tf
import numpy as np
import utils
import config
import os, glob
import scipy.misc
import cv2

from argparse import ArgumentParser
#from face_warp import *
from scipy import ndimage
import imageio
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

slim = tf.contrib.slim


def build_parser():
    parser = ArgumentParser()
    models_str = ' / '.join(config.model_zoo)
    parser.add_argument('--model', help=models_str, required=True)
    parser.add_argument('--name', help='default: name=model')
    parser.add_argument('--sample_size','-N',help='# of samples.It should be a square number. (default: 16)',default=16,type=int)

    return parser

def pre_precess_LR(im, crop_size):#将测试图像人脸框出，resize成64的HR和16的LR
    output_height, output_width = crop_size
    h, w = im.shape[:2]
    if h < output_height and w < output_width:
        raise ValueError("image is small")

    offset_h = int((h - output_height) / 2)
    offset_w = int((w - output_width) / 2)
    im = im[offset_h:offset_h+output_height, offset_w:offset_w+output_width, :]
    LR = scipy.misc.imresize(im,[16,16])
    HR = scipy.misc.imresize(im,[64,64])
    # scipy.misc.imsave('119956_lr.jpg', im_LR)
    # scipy.misc.imsave('119956_hr.jpg', im_HR)
    im_LR = np.reshape(LR, [-1, 16, 16, 3])
    im_HR = np.reshape(HR, [-1, 64, 64, 3])


    return im_LR,im_HR,LR,HR
#prec msc
def pre_precess_msc(im, crop_size):

    #im = im[60:60+128, 25:25+128, :]
    
    #LR = scipy.misc.imresize(im,[16,16])
    
    #LR = scipy.misc.imresize(im,[128,128])

    #LR = scipy.misc.imresize(im,[16,16],interp='bicubic')
    LR = scipy.misc.imresize(im,[128,128])
    HR = LR
    # HR = scipy.misc.imresize(im,[64,64])
    # scipy.misc.imsave('119956_lr.jpg', im_LR)
    # scipy.misc.imsave('119956_hr.jpg', im_HR)
    im_LR = np.reshape(LR, [-1, 128, 128, 3])
    im_HR = np.reshape(HR, [-1, 128, 128, 3])


    return im_LR,im_HR,LR,HR
def pre_precess_lan(im, crop_size):

    #im = im[152-64:152+64, 122-64:122+64, :]
    LR = scipy.misc.imresize(im,[16,16])
    HR = scipy.misc.imresize(im,[64,64])
    # scipy.misc.imsave('119956_lr.jpg', im_LR)
    # scipy.misc.imsave('119956_hr.jpg', im_HR)
    im_LR = np.reshape(LR, [-1, 16, 16, 3])
    im_HR = np.reshape(HR, [-1, 64, 64, 3])

    return im_LR,im_HR,LR,HR
def sample_z(shape):
    return np.random.uniform(-0.1,0.1,size=shape)

    return im_LR,im_HR,LR,HR
def pre_precess_lan_20(im, mean_value, std_value):

    # 105.687841643 2660.10447792 std = 51.57, mean = 105.68
    #im = im[152-64:152+64, 122-64:122+64, :]
    # mean_value = 105.69 - factor
    # std_value = 60 - factor
    # mean_face = scipy.misc.imread('/home/xujinchang/share/project/GAN/Face-hallucination-with-tiny-images/mean_face.jpg',mode='RGB')
    # print "mean: std: ", np.mean(mean_face), np.std(mean_face)
    LR = scipy.misc.imresize(im,[16,16])
    # LR = LR + 0.2*mean_face

    LR = mean_value + std_value * (LR - np.mean(LR)) / np.sqrt(np.var(LR))
    # LR = (LR - mean_LR) * 1.5 + 1.2*mean_LR
    LR[LR > 255] = 255
    LR[LR < 0] = 0
    HR = scipy.misc.imresize(im,[64,64])
    # scipy.misc.imsave('119956_lr.jpg', im_LR)
    # scipy.misc.imsave('119956_hr.jpg', im_HR)
    im_LR = np.reshape(LR, [-1, 16, 16, 3])
    im_HR = np.reshape(HR, [-1, 64, 64, 3])
    return im_LR,im_HR,LR,HR

def pre_precess_lan_5(im, factor1, factor2):

    # 105.687841643 2660.10447792 std = 51.57, mean = 105.68
    #im = im[152-64:152+64, 122-64:122+64, :]
    # mean_value = 105.69 - factor
    # std_value = 60 - factor
    # mean_face = scipy.misc.imread('/home/xujinchang/share/project/GAN/Face-hallucination-with-tiny-images/mean_face.jpg',mode='RGB')
    # print "mean: std: ", np.mean(mean_face), np.std(mean_face)
    LR = scipy.misc.imresize(im,[16,16])
    # LR = LR + 0.2*mean_face
    LR = (LR - mean_LR) * factor1 + factor2*mean_LR
    LR[LR > 255] = 255
    LR[LR < 0] = 0
    HR = scipy.misc.imresize(im,[64,64])
    # scipy.misc.imsave('119956_lr.jpg', im_LR)
    # scipy.misc.imsave('119956_hr.jpg', im_HR)
    im_LR = np.reshape(LR, [-1, 16, 16, 3])
    im_HR = np.reshape(HR, [-1, 64, 64, 3])
    return im_LR,im_HR,LR,HR

def pre_precess_mean(im, mean_face, factor):

    # 105.687841643 2660.10447792 std = 51.57, mean = 105.68
    #im = im[152-64:152+64, 122-64:122+64, :]
    # mean_value = 105.69 - factor
    # std_value = 60 - factor
    # 
    # print "mean: std: ", np.mean(mean_face), np.std(mean_face)
    LR = scipy.misc.imresize(im,[16,16])
    LR = (LR + factor*mean_face) / (1+factor)
    # LR = (LR - mean_LR) * factor1 + factor2*mean_LR
    LR[LR > 255] = 255
    LR[LR < 0] = 0
    HR = scipy.misc.imresize(im,[64,64])
    # scipy.misc.imsave('119956_lr.jpg', im_LR)
    # scipy.misc.imsave('119956_hr.jpg', im_HR)
    im_LR = np.reshape(LR, [-1, 16, 16, 3])
    im_HR = np.reshape(HR, [-1, 64, 64, 3])
    return im_LR,im_HR,LR,HR

def pre_precess_lfw_8(im, landmarks):

    #im = im[152-64:152+64, 122-64:122+64, :]
    im = face_warp_main(im, landmarks, "origin")
    LR = scipy.misc.imresize(im,[16,16])
    HR = scipy.misc.imresize(im,[64,64])
    # scipy.misc.imsave('119956_lr.jpg', im_LR)
    # scipy.misc.imsave('119956_hr.jpg', im_HR)
    im_LR = np.reshape(LR, [-1, 16, 16, 3])
    im_HR = np.reshape(HR, [-1, 64, 64, 3])

    return im_LR,im_HR,LR,HR
def sample_z(shape):
    return np.random.normal(size=shape)


def get_all_checkpoints(ckpt_dir, force=False):
    '''
    When the learning is interrupted and resumed, all checkpoints can not be fetched with get_checkpoint_state
    (The checkpoint state is rewritten from the point of resume).
    This function fetch all checkpoints forcely when arguments force=True.
    '''

    if force:
        ckpts = os.listdir(ckpt_dir) # get all fns
        ckpts = list(map(lambda p: os.path.splitext(p)[0], ckpts)) # del ext
        ckpts = set(ckpts) # unique
        ckpts = filter(lambda x: x.split('-')[-1].isdigit(), ckpts) # filter non-ckpt
        ckpts = sorted(ckpts, key=lambda x: int(x.split('-')[-1])) # sort
        ckpts = list(map(lambda x: os.path.join(ckpt_dir, x), ckpts)) # fn => path
    else:
        ckpts = tf.train.get_checkpoint_state(ckpt_dir).all_model_checkpoint_paths

    return ckpts


def eval(model, name, sample_shape=[1,1], load_all_ckpt=True):
    if name == None:
        name = model.name
    dir_name = 'eval/' + name
    if tf.gfile.Exists(dir_name):
        tf.gfile.DeleteRecursively(dir_name)
    tf.gfile.MakeDirs(dir_name)
    dir_name_sr = dir_name + '/sr'
    dir_name_lr = dir_name + '/lr'
    dir_name_hr = dir_name + '/hr'
    tf.gfile.MakeDirs(dir_name_sr)
    tf.gfile.MakeDirs(dir_name_lr)
    tf.gfile.MakeDirs(dir_name_hr)

    # training=False => generator only
    restorer = tf.train.Saver(slim.get_model_variables())

    config = tf.ConfigProto()
    # best_gpu = utils.get_best_gpu()
    config.gpu_options.allow_growth = True # Works same as CUDA_VISIBLE_DEVICES!
    with tf.Session(config=config) as sess:
        ckpts = get_all_checkpoints('/home/dl1/zmm/SR-with-tinyface/checkpoints/' + name, force=load_all_ckpt)
        restorer.restore(sess, ckpts[-1])
        size = sample_shape[0] * sample_shape[1]
        #fp = open('/home/xujinchang/share/project/GAN/tf_WGAN_GP/msc_test','r')
        # fp = open('/home/xujinchang/share/project/GAN/tf_WGAN_GP/celeA_List/test_lan_100','r')
        # fp = open('/home/xujinchang/share/project/GAN/tf_WGAN_GP/lan_paper_part','r')
        # path_image = '/home/safe_data_dir_2/guoyu/jpg_lanzhou_crop/'
        # path_image = "/home/tmp_data_dir/zhaoyu/CelebA/img_align_celeba/"
        #path_image = '/home/xujinchang/share/project/GAN/tf_WGAN_GP/m.0bh0sn/'
        # path_image = "/home/xujinchang/100_128face/"
        # image_list  = []
        # for line in fp.readlines():
        #     image_list.append(line.strip().split(' ')[0])
        # count = 0
        #fp = open('F:/chenlong/code/Face-hallucination-with-tiny-images-master/test_result','r')
        # fp = open('/home/xujinchang/share/project/GAN/tf_WGAN_GP/mean_face/mean_face_msc','r')
        # fp = open('/home/xujinchang/share/project/GAN/tf_WGAN_GP/lan_t1','r')
        #fp = open('C:/Users/X/Desktop/real/')

        path_image = '/home/dl1/zmm/wildface_CROP/'

        # path_image = '/home/xujinchang/share/project/GAN/Face-hallucination-with-tiny-images/'

        ##image_list  = [ '10000.jpg','10001.jpg','10002.jpg','10003.jpg','10004.jpg','10005.jpg','10006.jpg','10007.jpg','10008.jpg','10009.jpg',
                        ##'10010.jpg','10011.jpg','10012.jpg','10013.jpg','10014.jpg','10015.jpg','10016.jpg','10017.jpg','10018.jpg','10019.jpg',
                        ##'10020.jpg','10021.jpg','10022.jpg','10023.jpg','10024.jpg','10025.jpg','10026.jpg','10027.jpg','10028.jpg','10029.jpg',
                        ##'10030.jpg','10031.jpg','10032.jpg','10033.jpg','10034.jpg','10035.jpg','10036.jpg','10037.jpg','10038.jpg','10039.jpg',
                        ##'10040.jpg','10041.jpg','10042.jpg','10043.jpg','10044.jpg','10045.jpg','10046.jpg','10047.jpg','10048.jpg','10049.jpg',
                        ##'10050.jpg','10051.jpg','10052.jpg','10053.jpg','10054.jpg','10055.jpg','10056.jpg','10057.jpg','10058.jpg','10059.jpg',
                        ##'10060.jpg','10061.jpg','10062.jpg','10063.jpg','10064.jpg','10065.jpg','10066.jpg','10067.jpg','10068.jpg','10069.jpg',
                        ##'10070.jpg','10071.jpg','10072.jpg','10073.jpg','10074.jpg','10075.jpg','10076.jpg','10077.jpg','10078.jpg','10079.jpg',
                        ##'10080.jpg','10081.jpg','10082.jpg','10083.jpg','10084.jpg','10085.jpg','10086.jpg','10087.jpg','10088.jpg','10089.jpg',
                        ##'10090.jpg','10091.jpg','10092.jpg','10093.jpg','10094.jpg','10095.jpg','10096.jpg','10097.jpg','10098.jpg','10099.jpg',
                        ##'10100.jpg','10101.jpg','10102.jpg','10103.jpg','10104.jpg','10105.jpg','10106.jpg','10107.jpg','10108.jpg','10109.jpg',
                        ##'10110.jpg','10111.jpg','10112.jpg','10113.jpg','10114.jpg','10115.jpg','10116.jpg','10117.jpg','10118.jpg','10119.jpg',
                        ##'10120.jpg','10121.jpg','10122.jpg','10123.jpg','10124.jpg','10125.jpg','10126.jpg','10127.jpg','10128.jpg','10129.jpg',
                        ##'10130.jpg','10131.jpg','10132.jpg','10133.jpg','10134.jpg','10135.jpg','10136.jpg','10137.jpg','10138.jpg','10139.jpg',
                        ##'10140.jpg','10141.jpg','10142.jpg','10143.jpg','10144.jpg','10145.jpg','10146.jpg','10147.jpg','10148.jpg','10149.jpg',
                        ##'10150.jpg','10151.jpg','10152.jpg','10153.jpg','10154.jpg','10155.jpg','10156.jpg','10157.jpg','10158.jpg','10159.jpg',
                        ##'10160.jpg','10161.jpg','10162.jpg','10163.jpg','10164.jpg','10165.jpg','10166.jpg','10167.jpg','10168.jpg','10169.jpg',
                        ##'10170.jpg','10171.jpg','10172.jpg','10173.jpg','10174.jpg','10175.jpg','10176.jpg','10177.jpg','10178.jpg','10179.jpg',
                        ##'10180.jpg','10181.jpg','10182.jpg','10183.jpg','10184.jpg','10185.jpg','10186.jpg','10187.jpg','10188.jpg','10189.jpg',
                        ##'10190.jpg','10191.jpg','10192.jpg','10193.jpg','10194.jpg','10195.jpg','10196.jpg','10197.jpg','10198.jpg','10199.jpg',]

        #image_list = ['276_cam1_1.jpg']
        #for line in fp.readlines():
             #image_list.append(line.strip().split(' ')[-1])
        image_list=[]
        for root,dirs,files in os.walk(path_image):
            image_list=files
        
        #with open('F:/chenlong/code/Face-hallucination-with-tiny-images-master/crop1/','r')as fp:
            #imglist = [line.strip().split() for line in fp]
        #image_list = [line[0] for line in imglist]
        # landmarks = [line[1:11] for line in imglist]
        # sum_mean = 0.0
        # sum_var = 0.0
        # factor  = 1
        # mean_face = np.zeros((1,64,64,3), dtype=np.float32)
        # mean_face2 = scipy.misc.imread('/home/xujinchang/share/project/GAN/Face-hallucination-with-tiny-images/mean_face_msc.jpg',mode='RGB')
        # mean_face1 = scipy.misc.imread('/home/xujinchang/share/project/GAN/Face-hallucination-with-tiny-images/mean_face_lans.jpg',mode='RGB')
        for j in range(len(image_list)):
            
            print("count: {}".format(j))
            image = image_list[j]
            print ("image: " ,path_image + image_list[j])
            im = scipy.misc.imread(path_image + image_list[j], mode='RGB')
            count = 0
            # mean_value = [np.mean(im), 115.54, 100.54, 120.54]
            # std_value = [np.std(im), 51.29, 60.22]
            # factor1 = [1]   
            #factor2 = [0,0.2,0.4,0.5,0.6,0.7,0.8,1.0,2.0]
    
            #for i in range(len(factor2)):
            #count += 1
            z_, gt, LR, HR = pre_precess_msc(im, 128)
            z_ = z_ / 127.5 - 1.0
            fake_samples = sess.run(model.fake_sample, {model.z: z_})
            fake_samples = (fake_samples + 1.) * 127.5
            merged_samples = utils.merge(fake_samples, size=sample_shape)
            fn = str(image_list[j]) #+ '.jpg'
            
            scipy.misc.imsave(os.path.join(dir_name_sr, fn), merged_samples)
            scipy.misc.imsave(os.path.join(dir_name_lr, fn), LR)
            scipy.misc.imsave(os.path.join(dir_name_hr, fn), HR)
                # count += 1
                # z_, gt, LR, HR = pre_precess_mean(im, mean_face2, factor2[i])
                # z_ = z_ / 127.5 - 1.0
                # fake_samples = sess.run(model.fake_sample, {model.z: z_})
                # fake_samples = (fake_samples + 1.) * 127.5
                # merged_samples = utils.merge(fake_samples, size=sample_shape)
                # fn = str(count) + '.jpg'
                # scipy.misc.imsave(os.path.join(dir_name_sr, fn), merged_samples)

            #to_gif(dir_name_sr)
            # z_, gt, LR, HR = pre_precess_lan(im, [128, 128])
            # z_ = z_ / 127.5 - 1.0
            # fake_samples = sess.run(model.fake_sample, {model.z: z_})
            # fake_samples = (fake_samples + 1.) * 127.5
            # merged_samples = utils.merge(fake_samples, size=sample_shape)
            # image_ = image.split('/')[-1]
            # fn = "sr_"+image_
            # scipy.misc.imsave(os.path.join(dir_name_sr, fn), merged_samples)
            # mean_face += HR
            # image = image_list[j]
            # fn_lr = "lr_"+image_
            # fn_hr = "hr_"+image_
            # scipy.misc.imsave(os.path.join(dir_name_lr, fn_lr),LR)
            # scipy.misc.imsave(os.path.join(dir_name_hr, fn_hr),HR)
            
            
            

                
        # for v in ckpts:
        #     count += 1
        #     # if count < 50: continue
        #     print("Evaluating {} ...".format(v))
        #     restorer.restore(sess, v)
        #     global_step = int(v.split('/')[-1].split('-')[-1])

        #     fake_samples = sess.run(model.fake_sample, {model.z: z_})

        #     # inverse transform: [-1, 1] => [0, 1]
        #     fake_samples = (fake_samples + 1.) / 2.
        #     merged_samples = utils.merge(fake_samples, size=sample_shape)
        #     fn = "{:0>5d}.jpg".format(global_step)
        #     scipy.misc.imsave(os.path.join(dir_name, fn), merged_samples)
        # print "mean: , var: ", sum_mean / len(image_list), sum_var / len(image_list)
        # mean_face = mean_face / len(image_list)
        # mean_face = utils.merge(mean_face, size=sample_shape)
        # mean_face = scipy.misc.imresize(mean_face,[16,16])
        # scipy.misc.imsave(os.path.join(dir_name_lr, 'mean_face'+'.jpg'),mean_face)

'''
You can create a gif movie through imagemagick on the commandline:
$ convert -delay 20 eval/* movie.gif
'''
def to_gif(dir_name='eval'):
    images = []
    im_list = []
    # for path in glob.glob('*.jpg'):
    #     im_list.append(path) 
    for i in range(9):
        im = scipy.misc.imread(dir_name+'/'+str(i+1)+'.jpg')
        im = scipy.misc.imresize(im,[256,256])
        images.append(im)

    # make_gif(images, dir_name + '/movie.gif', duration=10, true_image=True)
    imageio.mimsave('movie.gif', images, duration=0.4)

if __name__ == "__main__":
    parser = build_parser()
    FLAGS = parser.parse_args()
    FLAGS.model = FLAGS.model.upper()
    if FLAGS.name is None:
        FLAGS.name = FLAGS.model.lower()
    config.pprint_args(FLAGS)

    N = FLAGS.sample_size**0.5
    assert N == int(N), 'sample size should be a square number'
    model = config.get_model(FLAGS.model, FLAGS.name, training=False)
    eval(model, name=FLAGS.name, sample_shape=[1,1], load_all_ckpt=True)
