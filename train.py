import os
import sys
import argparse
import tensorflow as tf
import numpy as np
import dbread as db
from model import Pix2Pix
import scipy.misc
import cv2
import matplotlib.pyplot as flg
tf.set_random_seed(7777)
parser = argparse.ArgumentParser(description='Easy Implementation of Pix2Pix')

# parameters
parser.add_argument('--train', type=str, default='filelist.txt')
parser.add_argument('--original', type=str, default='original.txt')
parser.add_argument('--out_dir', type=str, default='./output')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=1)

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
    global_epoch = tf.Variable(0, trainable=False, name='global_step')
    global_epoch_increase = tf.assign(global_epoch, tf.add(global_epoch, 1))
    args = parser.parse_args()
   
    filelist_train = args.train
    filelist_original = args.original
    result_dir = args.out_dir + '/result'
    back_dir = args.out_dir + '/back'
    ckpt_dir = args.out_dir + '/checkpoint'

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
       
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
   
    total_epoch = args.epochs
    batch_size = args.batch_size
    database = db.DBreader(filelist_train, batch_size=batch_size, labeled=False, resize=[512, 512])
    databaseo = db.DBreader(filelist_original, batch_size=batch_size, labeled=False, resize=[512, 512])

    #for i in database:
    
    sess = tf.Session()
    model = Pix2Pix(sess, batch_size)
    saver = tf.train.Saver(tf.global_variables())

    ckpt = tf.train.get_checkpoint_state(ckpt_dir)

    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
    total_batch = database.total_batch
    epoch = sess.run(global_epoch)
    print(total_batch)
    lossd = []
    lossgan = []
    lossl1 = []
    losstotal = []

    while True:
        templossd= []
        templossgan = []
        templossl1 = []
        total = []

        if epoch == total_epoch:
            flg.plot(range(1, len(lossd) + 1), lossd, 'r',  label= 'loss_perc')
            flg.plot(range(1, len(lossgan) + 1), lossgan, 'g',  label = 'loss_Gan')
            flg.plot(range(1, len(lossl1) + 1), lossl1, 'b',  label = 'loss_L1')
            flg.legend(loc='upper right')
            flg.ylabel('loss')
            flg.xlabel('Number of epoch')
            flg.savefig('result.png')
            saver.save(sess, ckpt_dir + '/model_epoch'+str(epoch).zfill(3))
            break

        if epoch % 5 == 0 :
            saver.save(sess, ckpt_dir + '/model_epoch'+str(epoch).zfill(3))

        for step in range(total_batch):         
            img_target = normalize(databaseo.next_batch())
            img_input_ori = database.next_batch()
            img_input = normalize(img_input_ori)

          
            if epoch > -1:
                loss_D = model.train_discrim(img_input, img_target)         # Train Discriminator and get the loss value
                print('Epoch: [', epoch, '/', total_epoch, '], ', 'Step: [', step, '/', total_batch, '] D_loss: ', loss_D)

            loss_GAN, loss_L1, loss_perc = model.train_gen(img_input_ori, img_target)  # Train Generator and get the loss value
            templossd.append(loss_perc*0.1)
            templossgan.append(loss_GAN)
            templossl1.append(loss_L1*20)
            total.append(loss_GAN+loss_L1*20+loss_perc*0.1) 

          
            print('Epoch: [', epoch, '/', total_epoch, '], ', 'Step: [', step, '/', total_batch, '], G_loss_GAN: ', loss_GAN, ', G_loss_L1: ', loss_L1,', perc_loss: ', loss_perc, ', Total_loss: ', loss_GAN + loss_L1*20 + loss_perc*0.1)

           
            if (step % 1600 == 0):
                generated_samples = denormalize(model.sample_generator(img_input, batch_size=batch_size))
                img_target = denormalize(img_target)
                img_input = denormalize(img_input)

    

                img_for_vis = np.concatenate([img_input, generated_samples, img_target], axis=2)
                savepath = result_dir + '/output_' + 'EP' + str(epoch).zfill(3) + "_Batch" + str(step).zfill(6) + '.png'
                savepath2 = back_dir + '/output_' + 'EP' + str(epoch).zfill(3) + "_Batch" + str(step).zfill(6) + '(back).png'
          
                save_visualization(img_for_vis, (batch_size, 1), save_path=savepath)
                save_visualization2(generated_samples, (batch_size, 1), save_path=savepath2)



        lossd.append(np.mean(templossd))
        lossgan.append(np.mean(templossgan))
        lossl1.append(np.mean(templossl1))
        losstotal.append(np.mean(total))

        flg.plot(range(1, len(lossd) + 1), lossd, 'r',  label= 'PercLoss')
        flg.plot(range(1, len(lossgan) + 1), lossgan, 'g',  label = 'AdvLoss')
        flg.plot(range(1, len(lossl1) + 1), lossl1, 'b',  label = 'L1Loss')
        flg.plot(range(1, len(losstotal) + 1), losstotal, 'c',  label = 'TotalLoss')
        flg.legend(loc='upper right')
        flg.ylabel('Loss')
        flg.xlabel('Number of epoch')
        flg.savefig('result.png')
        flg.cla()

        epoch = sess.run(global_epoch_increase)
        #saver.save(sess, ckpt_dir + '/model_epoch'+str(epoch).zfill(3))


if __name__ == "__main__":
    main()
