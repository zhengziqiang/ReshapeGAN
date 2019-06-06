import os
import tensorflow as tf
from ops import *
from utils import *
from glob import glob
class StarGAN(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.checkpoint_dir = args.checkpoint_dir
        self.ch = args.ch
        self.n_res = args.n_res
        self.img_size = args.img_size
        self.img_ch = args.img_ch
    def generator_withface(self, x_init, trg_face, reuse=False, scope="generator_stage2"):
        channel = self.ch
        x = tf.concat([x_init, trg_face], axis=-1)
        with tf.variable_scope(scope, reuse=reuse):
            x = conv(x, channel, kernel=7, stride=1, pad=3, use_bias=False, scope='conv')
            x = instance_norm(x, scope='ins_norm')
            x = relu(x)
            # Down-Sampling
            for i in range(2):
                x = conv(x, channel * 2, kernel=4, stride=2, pad=1, use_bias=False, scope='conv_' + str(i))
                x = instance_norm(x, scope='down_ins_norm_' + str(i))
                x = relu(x)
                channel = channel * 2

            # Bottleneck 6 layers
            for i in range(self.n_res):
                x = resblock(x, channel, use_bias=False, scope='resblock_' + str(i))

            # Up-Sampling
            for i in range(2):
                x = deconv(x, channel // 2, kernel=4, stride=2, use_bias=False, scope='deconv_' + str(i))
                x = instance_norm(x, scope='up_ins_norm' + str(i))
                x = relu(x)
                channel = channel // 2
            x = conv(x, channels=3, kernel=7, stride=1, pad=3, use_bias=False, scope='G_logit')
            x = tanh(x)
            return x

    def build_model(self,args):
        """ Result Image """
        self.n_ref = args.n_ref  # how many reference images multiple generations
        self.input_image = tf.placeholder(tf.float32, [1, self.img_size, self.img_size, self.img_ch],
                                           name='input_image')  # Input Image

        self.ref_image = tf.placeholder(tf.float32, [1, self.img_size, self.img_size, self.img_ch],
                                           name='ref_image')  # Single reference images
        self.ref_images = tf.placeholder(tf.float32,
                                                [self.n_ref, 1, self.img_size, self.img_size, self.img_ch],
                                                name='ref_images')  # Multiple reference images
        self.fake_image= self.generator_withface(self.input_image,self.ref_image,reuse=False, scope="generator_stage2")

        self.x_fake_face_list = tf.map_fn(
            lambda x: self.generator_withface(self.input_image, x, reuse=True, scope="generator_stage2"),
            #only one stage, and the name does not matter
            self.ref_images, dtype=tf.float32)
    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def test_single(self,args):
        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        input_img= misc.imread(args.input_img, mode='RGB')
        ref_img = misc.imread(args.ref_img, mode='RGB')
        input_img = misc.imresize(input_img, [self.img_size, self.img_size*2])
        input_img = np.expand_dims(input_img, axis=0)
        input_img = normalize(input_img)
        input_img = np.asarray(input_img)

        ref_img = misc.imresize(ref_img, [self.img_size, self.img_size*2])
        ref_img = np.expand_dims(ref_img, axis=0)
        ref_img = normalize(ref_img)
        ref_img = np.asarray(ref_img)
        fake_face = self.sess.run([self.fake_image],
                                       feed_dict={self.input_image: input_img[:,:,:self.img_size,:], self.ref_image: ref_img[:,:,self.img_size:,:]})
        merge_img=np.concatenate([input_img[:,:,:self.img_size,:],ref_img[:,:,:self.img_size,:],fake_face[0]],axis=0)
        save_images(merge_img, [1, 3], args.result_img)

    def test_multiple(self,args):
        self.result_dir = args.result_dir
        self.input_dir = args.input_dir
        self.n_ref = args.n_ref  # how many reference images multiple generations
        tf.global_variables_initializer().run()
        test_path = self.input_dir
        check_folder(test_path)
        test_files = glob(os.path.join(test_path, '*.*'))
        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        check_folder(self.result_dir)
        image_folder = os.path.join(self.result_dir, 'images')
        check_folder(image_folder)
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        for sample_file in test_files:
            p,n=os.path.split(sample_file)
            nn=n.split(".")
            name=nn[0]
            self.save_fig_dir = os.path.join(image_folder,name)
            check_folder(self.save_fig_dir)
            print("Processing image: " + sample_file)
            input_img,face_imgs,landmark=load_test_data(sample_file,self.n_ref,size=self.img_size)
            input_img=np.asarray(input_img)
            face_imgs=np.asarray(face_imgs)
            concat_face=face_imgs[:,0,:,:,:]
            image_path_face = os.path.join(self.save_fig_dir, '{}_face.jpg'.format(name))
            fake_face_list = self.sess.run([self.x_fake_face_list], feed_dict={self.input_image: input_img,self.ref_images:landmark})
            fake_face_list = np.transpose(fake_face_list[0], axes=[1, 0, 2, 3, 4])[0]  # [bs, c_dim, h, w, ch]
            fake_face_list = np.concatenate([input_img,concat_face,fake_face_list], axis=0)
            save_images(fake_face_list,[1, self.n_ref*2 + 1],image_path_face)
