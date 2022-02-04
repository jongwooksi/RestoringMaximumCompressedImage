import cv2
import os
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
import scipy.misc
import numpy as np
import scipy.signal
import scipy.ndimage
import vifp

'''
back_pix2pix                     : 20.537, 0.586, 0.211
back_ipiu                        : 20.623, 0.604, 0.222
back_netmodify                   : 21.445, 0.665, 0.255
back_netmodify_vgg16             : 21.558, 0.666, 0.255
back_netmodify_vgg16_stopD       : 21.617, 0.670, 0.257
back_netmodify_vgg16_L1w10       : 21.573, 0.670, 0.257
back_netmodify_vgg16_L1w10_stopD : 21.675, 0.673, 0.259
back_netmodify_vgg16_L1w20       : 21.596, 0.671, 0.257
back_netmodify_vgg16_L1w20_stopD : 21.680, 0.675, 0.261
'''

def cal_ssim(grayA, grayB):
    (score, diff) = structural_similarity(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    return score

def valuelist(size):
    object = list()
    for i in range(size):
        object.append( list() ) 
    return object

originalPath = '/home/jwsi/PIX2PIX/database/face/원본파일_test/'
generatePath = '/home/jwsi/PIX2PIX/output_test/back98_pix2pix/'
generatePath2 = '/home/jwsi/PIX2PIX/output_test/back_netmodify_vgg16_stopD/'

file_style = os.listdir(generatePath)
file_style.sort()

file_content = os.listdir(originalPath)
file_content.sort()


count = 0

psnr_content = [0 for i in range(3)]
ssim_content = [0 for i in range(3)]
vifp_content = [0 for i in range(3)]
hist_content1 = []
hist_content2 = []
hist_content3 = []
hist_content4 = []
size_original = 0
size_generate = 0

for i in range(len(file_style)):
   
    img1 = cv2.imread(originalPath+file_content[i])
    img3 = cv2.imread(generatePath+ file_style[i])
    size_original += os.path.getsize(originalPath+file_content[i])
    size_generate += os.path.getsize(generatePath+ file_style[i])
 
    psnr_content[0] += cv2.PSNR(img1, img1)
    psnrvalue = cv2.PSNR(img1, img3)
    psnr_content[1] += psnrvalue

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

    ssim_content[0] += cal_ssim(img1, img1)
    ssimvalue =  cal_ssim(img1, img3)
    ssim_content[1] +=ssimvalue
 
    ref = scipy.misc.imread(originalPath+file_content[i], flatten=True).astype(np.float32)
    dist2 = scipy.misc.imread(generatePath+ file_style[i], flatten=True).astype(np.float32)
 
    vifp_content[0]+=vifp.vifp_mscale(ref, ref)
    vifpvalue = vifp.vifp_mscale(ref, dist2)
    vifp_content[1]+=vifpvalue
    '''
    img1 = cv2.imread(originalPath+file_content[i])
    img3 = cv2.imread(generatePath+ file_style[i])
 
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hist1 = cv2.calcHist([hsv1], [0,1], None, [180,256], [0,180,0, 256]) 
    original= cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)

    hsv2 = cv2.cvtColor(img3, cv2.COLOR_BGR2HSV)
    hist2 = cv2.calcHist([hsv2], [0,1], None, [180,256], [0,180,0, 256]) 
    genImage = cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)

    methods = {'CORREL' :cv2.HISTCMP_CORREL, 'CHISQR':cv2.HISTCMP_CHISQR, 
           'INTERSECT':cv2.HISTCMP_INTERSECT,
           'BHATTACHARYYA':cv2.HISTCMP_BHATTACHARYYA}

    tempret=[0,0,0,0]

    for j, (name, flag) in enumerate(methods.items()):
        
        ret = cv2.compareHist(original, genImage, flag)
        if flag == cv2.HISTCMP_CORREL:
            tempret[0] = ret
            hist_content1.append(ret)

        elif flag == cv2.HISTCMP_CHISQR:
            tempret[1] = ret
            hist_content2.append(ret)
  
        elif flag == cv2.HISTCMP_INTERSECT: 
            ret = ret/np.sum(original)    
            tempret[2] = ret    
            hist_content3.append(ret)

        elif flag == cv2.HISTCMP_BHATTACHARYYA:
            tempret[3] = ret
            hist_content4.append(ret)
      

        #print("img%d:%7.2f"% (i+1 , ret), end='\t')
        '''
    print(file_content[i],psnrvalue, ssimvalue,vifpvalue )
    count += 1



contentloss = [x/count for x in psnr_content]
contentloss2 = [x/count for x in ssim_content]
vifploss = [x/count for x in vifp_content]


for i in range(3):  
    print("{:>7.3f}".format(contentloss[i]), end= "        ")
print()

for i in range(3):  
    print("{:>7.3f}".format(contentloss2[i]), end= "        ")
print()

for i in range(3):  
    print("{:>7.3f}".format(vifploss[i]), end= "        ")
print()


print(size_original, size_generate, size_generate/size_original)
#print(sum(hist_content1)/count)
#print(sum(hist_content2)/count)
#print(sum(hist_content3)/count)
#print(sum(hist_content4)/count)


