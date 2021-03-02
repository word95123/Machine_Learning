import os, cv2, numpy as np, json, re, base64
import random
import csv

class predict():
    def __init__(self):
        self.filenames = []
        self.normalImg = []
        self.normalMask = []
        self.abnormalImg = []
        self.abnormalMask = []
        self.totalImg = None
        self.totalMask = None
        self.x_train = None
        self.y_train = None
        self.x_valid = None
        self.y_valid = None

    def load_file(self, normalFile = '', abnormalFile = '', saveImg = False):
        [os.remove('output/'+i) for i in os.listdir('output')]
        #normal
        if normalFile != '':
            for file in os.listdir(normalFile):
                if re.split('[.]', file)[-1] in ['jpg']:
                    name='.'.join(re.split('[.]', file)[:-1])
                    self.filenames.append(name)
                    img=cv2.imread('./' + normalFile + '/' + name + '.jpg', cv2.IMREAD_GRAYSCALE)
                    img = img[60:580, 100:750]
                    img=cv2.resize(img, (416, 416))

                    img = img.reshape(img.shape[0], img.shape[1], 1)
                    self.normalImg.append(img)
                    
                    masks=np.zeros_like(img)
                    self.normalMask.append(masks)

                    if saveImg:
                        cv2.imwrite('output/'+name+'.jpg',np.concatenate((img,masks),1))
        #abnormal
        if abnormalFile != '':
            for file in os.listdir(abnormalFile):
                if re.split('[.]', file)[-1] in ['json']:
                    name='.'.join(re.split('[.]', file)[:-1])
                    self.filenames.append(name)
                    data=json.loads(open(abnormalFile +'/'+name+'.json', 'r').read())
                    img=cv2.imdecode(np.frombuffer(base64.b64decode(data['imageData']), np.uint8), cv2.IMREAD_GRAYSCALE)
                    
                    masks=np.zeros_like(img)
                    for shape in data['shapes']:
                        pts=np.array(shape['points'])
                        cv2.fillPoly(img=masks, pts=np.expand_dims(np.array(pts), 0).astype('int'), color=(255,))
                    img = img[60:580, 100:750]
                    img=cv2.resize(img, (416, 416))

                    img = img.reshape(img.shape[0], img.shape[1], 1)
                    self.abnormalImg.append(img)
                    
                    masks = masks[60:580, 100:750]
                    masks=np.expand_dims(cv2.resize(masks,(416,416)), 2)
                    self.abnormalMask.append(masks)

                    if saveImg:
                        cv2.imwrite('output/'+name+'.jpg', np.concatenate((img,masks),1))


    def data_process(self, randomSeed = 3210):
        normalImg = np.array(self.normalImg).copy()
        normalMask = np.array(self.normalMask).copy()
        abnormalImg = np.array(self.abnormalImg).copy()
        abnormalMask = np.array(self.abnormalMask).copy()
        random.seed(randomSeed)

        if len(self.normalImg) != 0:
            #normal
            normalShuffle = list(zip(self.normalImg, self.normalMask))
            random.shuffle(normalShuffle)
            self.normalImg, self.normalMask = zip(*normalShuffle)

            self.normalImg = np.array(self.normalImg)
            self.normalMask = np.array(self.normalMask)
            print(self.normalImg.shape, self.normalMask.shape)
            print(self.normalImg.dtype, self.normalMask.dtype)

            normalImgValid = self.normalImg[:int(len(self.normalImg)*0.1)]
            normalMaskValid = self.normalMask[:int(len(self.normalMask)*0.1)]

            
            normalImgTrain = self.normalImg[int(len(self.normalImg)*0.1):]
            normalMaskTrain = self.normalMask[int(len(self.normalMask)*0.1):]


        if len(self.abnormalImg) != 0:
            #abnormal
            abnormalShuffle = list(zip(self.abnormalImg, self.abnormalMask))
            random.shuffle(abnormalShuffle)
            self.abnormalImg, self.abnormalMask = zip(*abnormalShuffle)

            self.abnormalImg = np.array(self.abnormalImg)
            self.abnormalMask = np.array(self.abnormalMask)
            print(self.abnormalImg.shape, self.abnormalMask.shape)
            print(self.abnormalImg.dtype, self.abnormalMask.dtype)

            abnormalImgValid = self.abnormalImg[:int(len(self.abnormalImg)*0.1)]
            abnormalMaskValid = self.abnormalMask[:int(len(self.abnormalMask)*0.1)]

            abnormalImgTrain = self.abnormalImg[int(len(self.abnormalImg)*0.1):]
            abnormalMaskTrain = self.abnormalMask[int(len(self.abnormalMask)*0.1):]


        if len(self.normalImg) != 0 and len(self.abnormalImg) != 0:
            #merge data
            self.totalImg = np.vstack((normalImg, abnormalImg)).copy()
            self.totalMask = np.vstack((normalMask, abnormalMask)).copy()
            

            self.x_valid = np.vstack((normalImgValid, abnormalImgValid))
            self.y_valid = np.vstack((normalMaskValid, abnormalMaskValid))

            self.x_train = np.vstack((normalImgTrain, abnormalImgTrain))
            self.y_train = np.vstack((normalMaskTrain, abnormalMaskTrain))
        elif len(self.normalImg) != 0:
            self.totalImg = normalImg.copy()
            self.totalMask = normalMask.copy()

            self.x_valid = normalImgValid.copy()
            self.y_valid = normalMaskValid.copy()

            self.x_train = normalImgTrain.copy()
            self.y_train = normalMaskTrain.copy()
        elif len(self.abnormalImg) != 0:
            self.totalImg = abnormalImg.copy()
            self.totalMask = abnormalMask.copy()

            self.x_valid = abnormalImgValid.copy()
            self.y_valid = abnormalMaskValid.copy()

            self.x_train = abnormalImgTrain.copy()
            self.y_train = abnormalMaskTrain.copy()
        print(len(self.abnormalImg))

    def normalize_data(self):
        self.x_train = self.x_train.astype('float32')/255.0
        self.y_train = self.y_train.astype('float32')/255.0
        self.x_valid = self.x_valid.astype('float32')/255.0
        self.y_valid = self.y_valid.astype('float32')/255.0

        print(self.x_train.shape, self.y_train.shape)
        print(self.x_valid.shape, self.y_valid.shape)
        print(self.x_train.dtype, self.y_train.dtype)
        print(self.x_valid.dtype, self.y_valid.dtype)


    def predicting(self, conModel, MultilayerModel):
        os.environ['SM_FRAMEWORK']='tf.keras'
        try:
            import segmentation_models as sm
        except:
            print('should "pip install segmentation_models"')
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, UpSampling2D, Dropout
        from tensorflow.keras.optimizers import Adam
        #model1
        backbone='resnet34'
        model = sm.Unet(backbone, classes=1, input_shape=(416, 416, 1), activation='sigmoid', encoder_weights=None)
        model.compile(
            Adam(lr=1e-4),
            loss=sm.losses.bce_jaccard_loss,
            metrics=[sm.metrics.iou_score],
        )

        model.load_weights(conModel)
        #iou=model.evaluate(self.x_valid, self.y_valid, batch_size=16, verbose=0)[-1]

        [os.remove('pred/'+i) for i in os.listdir('pred')]
        pred=model.predict(self.abnormalImg.astype('float32')/255.0)
        pred=np.where(pred>0.5, 255, 0).astype('uint8')


        for i in range(len(pred)):
            im=cv2.cvtColor(self.abnormalImg[i][:,:,0], cv2.COLOR_GRAY2BGR)
            show=im.copy()
            for j in range(1):
                ma=im.copy()
                ma[np.where(self.abnormalImg[i][:,:,j]==255)]=np.array([0,255,0], dtype='uint8')
                show=np.concatenate((show, ma),1)
                
            for j in range(1):
                ma=im.copy()
                ma[np.where(pred[i][:,:,j]==255)]=np.array([0,255,0], dtype='uint8')
            pic = np.concatenate((show, ma), 1)
            cv2.imwrite('pred/'+str(i)+'.jpg', pic)
        
        multiMask = []
        pred = model.predict(self.totalImg.astype('float32')/255.0)
        pred = np.where(pred>0.5, 255, 0).astype('uint8')

        for i in range(len(pred)):
            im=cv2.cvtColor(self.totalImg[i][:,:,0], cv2.COLOR_GRAY2BGR)
            show=im.copy()
            ma=im.copy()
            ma[np.where(self.totalMask[i][:,:,0]==255)]=np.array([0,255,0], dtype='uint8')
            show=np.concatenate((show, ma),1)
            ma=im.copy()
            ma[np.where(pred[i][:,:,0]==255)]=np.array([0,255,0], dtype='uint8')
            show=np.concatenate((show, ma),1)
            #merge mask
            if i < len(self.normalImg):
                mask1 = self.totalMask[i][:,:,0]
                mask2 = pred[i][:,:,0]
            else:
                mask1 = self.totalMask[i][:,:,0]
                mask2 = np.zeros_like(mask1)
            mask1 = np.expand_dims(mask1, axis=2)
            mask2 = np.expand_dims(mask2, axis=2)

            multiMask.append(np.concatenate((mask1, mask2), 2))
            
            
            #cv2.imwrite('output/'+str(i)+'.jpg', show)

        multiMask = np.array(multiMask)

        for i in range(len(multiMask)):
            cv2.imwrite('output_multimask/'+str(i)+'.jpg', multiMask[i][:,:,0])
            cv2.imwrite('output_multimask/'+str(i)+'_.jpg', multiMask[i][:,:,1])

        x_valid = self.totalImg[:int(len(self.totalImg)*0.1)]
        y_valid = multiMask[:int(len(multiMask)*0.1)]

        x_train = self.totalImg[int(len(self.totalImg)*0.1):]
        y_train = multiMask[int(len(multiMask)*0.1):]


        x_train = x_train.astype('float32')/255.0
        y_train = y_train.astype('float32')/255.0
        x_valid = x_valid.astype('float32')/255.0
        y_valid = y_valid.astype('float32')/255.0
        #model2
        backbone='resnet50'
        model = sm.Unet(backbone, classes=2, input_shape=(416, 416, 1), activation='sigmoid', encoder_weights=None)
        model.compile(
            Adam(lr=1e-4),
            loss=sm.losses.bce_jaccard_loss,
            metrics=[sm.metrics.iou_score],
        )

        model.load_weights(MultilayerModel)

        iou=model.evaluate(x_valid, y_valid, batch_size=16, verbose=0)[-1]

        [os.remove('multilayer_pred/'+i) for i in os.listdir('multilayer_pred')]
        pred=model.predict(self.totalImg.astype('float32')/255.0)
        pred=np.where(pred>0.5, 255, 0).astype('uint8')

        pic_proportion = []
        for i in range(len(pred)):
            im=cv2.cvtColor(self.totalImg[i][:,:,0], cv2.COLOR_GRAY2BGR)
            show=im.copy()
            '''
            ma=im.copy()
            ma[np.where(multiMask[i][:,:,0]==255)]=np.array([0,255,0], dtype='uint8')
            show=np.concatenate((show, ma),1)
            '''
            ma=im.copy()
            ma[np.where(pred[i][:,:,0]==255)]=np.array([0,255,0], dtype='uint8')
            show=np.concatenate((show, ma),1)
            proportion = 0
            #check proportion
            for p in range(len(pred[i])):
                for q in range(len(pred[i])):
                    if pred[i][p,q,0]==255:
                        proportion += 1
            proportion = proportion/(416*416)

            pic_proportion.append(proportion)
            '''
            ma=im.copy()
            ma[np.where(multiMask[i][:,:,1]==255)]=np.array([255,0,0], dtype='uint8')
            show=np.concatenate((show, ma),1)

            ma=im.copy()
            ma[np.where(pred[i][:,:,1]==255)]=np.array([255,0,0], dtype='uint8')
            show=np.concatenate((show, ma),1)
            '''
            cv2.imwrite('multilayer_pred/'+self.filenames[i]+'.jpg', show)


        with open('proportion.csv','w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['檔名', '比例'])
            for p in range(len(pic_proportion)):
                writer.writerow([str(self.filenames[p]),str(pic_proportion[p])])

    

if __name__ == "__main__":
    predict = predict()
    predict.load_file(normalFile = '', abnormalFile = 'predictions/abnormal_data/', saveImg=True)
    predict.data_process()
    predict.normalize_data()
    print('predict')
    predict.predicting(conModel = 'model_ver2.1.h5', MultilayerModel = 'model_ver3.0.h5')
    print('end')
    pass