import os
import sys
import argparse
import tensorflow as tf
import numpy as np
import dbread as db
from model import Pix2Pix
import scipy.misc
import cv2

parser = argparse.ArgumentParser(description='Easy Implementation of Pix2Pix')

# parameters
parser.add_argument('--test', type=str, default='testlist.txt')
parser.add_argument('--out_dir', type=str, default='./output_test')
parser.add_argument('--ckpt_dir', type=str, default='./output/checkpoint')
parser.add_argument('--visnum', type=int, default=1)

def normalize(im):
    return im * (2.0 / 255.0) - 1


def denormalize(im):
    return (im + 1.) / 2.




# Function for save the generated result
def save_visualization2(X, nh_nw, save_path='./vis/sample.jpg'):
    nh, nw = nh_nw

    h, w = X.shape[1], X.shape[2]
    img = np.zeros((h * nh, w * nw, 3))

    for n, x in enumerate(X):
        j = int(n / nw)
        i = int(n % nw)
        img[j * h:j * h + h, i * w:i * w + w, :] = x

    
    img = cv2.resize(img, (128,128),  interpolation=cv2.INTER_AREA)

    scipy.misc.imsave(save_path, img)



def save_visualization(X, nh_nw, save_path='./vis/sample.jpg'):
    nh, nw = nh_nw

    h, w = X.shape[1], X.shape[2]
    img = np.zeros((h * nh, w * nw, 3))

    for n, x in enumerate(X):
        j = int(n / nw)
        i = int(n % nw)
        img[j * h:j * h + h, i * w:i * w + w, :] = x

   

    scipy.misc.imsave(save_path, img)



def main():
    args = parser.parse_args()

    filelist_test = args.test
    result_dir = args.out_dir + '/result'
    ckpt_dir = args.ckpt_dir
    back_dir = args.out_dir + '/back'

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    batch_size = args.visnum

    database = db.DBreader(filelist_test, batch_size=batch_size, labeled=False, resize=[512, 512], shuffle=False)
  #  databaseo = db.DBreader(filelist_toriginal, batch_size=batch_size, labeled=False, resize=[256, 256], suffle=False)

    sess = tf.Session()
    model = Pix2Pix(sess, batch_size)

    saver = tf.train.Saver(tf.global_variables())

    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sys.exit("There is no trained model")

    total_batch = database.total_batch

    print('Generating...')
    for step in range(total_batch):
        #img_input, img_target = split_images(database.next_batch(), direction)
        #img_target = normalize(databaseo.next_batch())
        img_input = normalize(database.next_batch())


        generated_samples = denormalize(model.sample_generator(img_input, batch_size=batch_size))
        #img_target = denormalize(img_target)
        img_input = denormalize(img_input)

        img_for_vis = np.concatenate([img_input, generated_samples], axis=2)
        savepath = result_dir + '/output_' + "_Batch" + str(step).zfill(6) + '.png'
        savepath2 = back_dir + '/output_'  + "_Batch" + str(step).zfill(6) + '(back).png'
          
        #save_visualization(img_for_vis, (batch_size, 1), save_path=savepath)
        save_visualization2(generated_samples, (batch_size, 1), save_path=savepath2)


    print('finished!!')

if __name__ == "__main__":
    main()
