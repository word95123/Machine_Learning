import os, cv2, numpy as np, json, re, base64
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class train():

    def __init__(self):
        self.filenames = []
        self.normalImg = []
        self.normalMask = []
        self.abnormalImg = []
        self.abnormalMask = []
        self.totalImg = self.totalMask = None
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
                    data=json.loads(open(abnormalFile+'/'+name+'.json', 'r').read())
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
        normalImg = self.normalImg.copy()
        normalMask = self.normalMask.copy()
        abnormalImg = self.abnormalImg.copy()
        abnormalMask = self.abnormalMask.copy()
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
            self.totalImg = np.array(normalImg).copy()
            self.totalMask = np.array(normalMask).copy()

            self.x_valid = normalImgValid.copy()
            self.y_valid = normalMaskValid.copy()

            self.x_train = normalImgTrain.copy()
            self.y_train = normalMaskTrain.copy()
        elif len(self.abnormalImg) != 0:
            self.totalImg = np.array(abnormalImg).copy()
            self.totalMask = np.array(abnormalMask).copy()

            self.x_valid = abnormalImgValid.copy()
            self.y_valid = abnormalMaskValid.copy()

            self.x_train = abnormalImgTrain.copy()
            self.y_train = abnormalMaskTrain.copy()
        


    def img_generator(self):
        data_gen_args = dict(rotation_range=10,
                            zoom_range=0.2,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            fill_mode='constant',
                            cval=0)
        image_datagen = ImageDataGenerator(**data_gen_args)
        mask_datagen = ImageDataGenerator(**data_gen_args)
        seed = 9487
        image_datagen.fit(self.x_train, augment=True, seed=seed)
        mask_datagen.fit(self.y_train, augment=True, seed=seed)
        image_generator = image_datagen.flow(self.x_train, seed=seed)
        mask_generator = mask_datagen.flow(self.y_train, seed=seed)
        # combine generators into one which yields image and masks
        train_generator = zip(image_generator, mask_generator)
        [os.remove('output_gens/'+i) for i in os.listdir('output_gens')]
        times=0
        for x_train_, y_train_ in train_generator:
            y_train_=y_train_.astype('uint8')
            for i in range(len(y_train_)):
                y_train_[i,:,0]=cv2.threshold(y_train_[i,:,0],1234,255,cv2.THRESH_OTSU|cv2.THRESH_BINARY)[-1]
            x_train_=x_train_.astype('uint8')
            for i in range(len(x_train_)):
                cv2.imwrite('output_gens/'+str(times)+'_'+str(i)+'.jpg', np.concatenate((x_train_[i],y_train_[i]),1))
            self.x_train=np.concatenate((self.x_train, x_train_), 0)
            self.y_train=np.concatenate((self.y_train, y_train_), 0)
            times+=1
            if times == 10:
                break
        
    def normalize_data(self):
        self.x_train=self.x_train.astype('float32')/255.0
        self.y_train=self.y_train.astype('float32')/255.0
        self.x_valid=self.x_valid.astype('float32')/255.0
        self.y_valid=self.y_valid.astype('float32')/255.0

        print(self.x_train.shape, self.y_train.shape)
        print(self.x_valid.shape, self.y_valid.shape)
        print(self.x_train.dtype, self.y_train.dtype)
        print(self.x_valid.dtype, self.y_valid.dtype)


    def training(self, conModel, MultilayerModel, steps = 100):
        trainingMask = False
        contiTrainingMask = False
        trainingMultiMask = True
        contiTrainingMultiMask = False

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
        #predict to produce multilayer_mask data
        model = sm.Unet(backbone, classes=1, input_shape=(416, 416, 1), activation='sigmoid', encoder_weights=None)
        model.compile(
            Adam(lr=1e-4),
            loss=sm.losses.bce_jaccard_loss,
            metrics=[sm.metrics.iou_score],
        )
        if trainingMask:
            if contiTrainingMask:
                model.load_weights(conModel)
            iou=model.fit(self.x_train, self.y_train, validation_data=(self.x_valid, self.y_valid), batch_size=8, epochs=1).history['val_iou_score'][0]
            model.save_weights('model.h5')
            for i in range(steps):
                iou_=model.fit(self.x_train, self.y_train, validation_data=(self.x_valid, self.y_valid), batch_size=8, epochs=1).history['val_iou_score'][0]
                if iou_>iou:
                    iou=iou_
                    model.save_weights('model.h5')
                    print('save', iou)
            model.load_weights('model.h5')  
        else:
            model.load_weights(conModel)
    
        multiMask = []
        pred=model.predict(self.totalImg.astype('float32')/255.0)
        pred=np.where(pred>0.5, 255, 0).astype('uint8')
        #merge mask1 and mask2 to multilayer_mask data
        for i in range(len(pred)):
            im=cv2.cvtColor(self.totalImg[i][:,:,0], cv2.COLOR_GRAY2BGR)
            show=im.copy()
            ma=im.copy()
            ma[np.where(self.totalMask[i][:,:,0]==255)]=np.array([0,255,0], dtype='uint8')
            show=np.concatenate((show, ma),1)
            ma=im.copy()
            ma[np.where(pred[i][:,:,0]==255)]=np.array([0,255,0], dtype='uint8')
            show=np.concatenate((show, ma),1)
            if i < len(self.normalImg):
                mask1 = self.totalMask[i][:,:,0]
                mask2 = pred[i][:,:,0]
            else:
                mask1 = self.totalMask[i][:,:,0]
                mask2 = np.zeros_like(mask1)
            mask1 = np.expand_dims(mask1, axis=2)
            mask2 = np.expand_dims(mask2, axis=2)

            multiMask.append(np.concatenate((mask1, mask2), 2))
            
            
            cv2.imwrite('output/'+str(i)+'.jpg', show)

        multiMask = np.array(multiMask)


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
         
        if trainingMultiMask:
            if contiTrainingMultiMask:
                model.load_weights(MultilayerModel)
            iou=model.fit(x_train, y_train, validation_data=(x_valid, y_valid), batch_size=8, epochs=1).history['val_iou_score'][0]
            for i in range(steps):
                iou_=model.fit(x_train, y_train, validation_data=(x_valid, y_valid), batch_size=8, epochs=1).history['val_iou_score'][0]
                if iou_>iou:
                    iou=iou_
                    model.save_weights('_model.h5')
                    print('save', iou)
            

if __name__ == "__main__":
    train = train()
    train.load_file(normalFile = 'predictions/normal/original_normal', abnormalFile = 'predictions/data', saveImg=True)
    train.data_process()
    #train.img_generator()
    train.normalize_data()
    train.training(conModel = './test/onlyAbnormal/model.h5', MultilayerModel = 'model_test.h5')
    print('end')
    pass