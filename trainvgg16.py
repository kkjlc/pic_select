import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D,Dropout,Input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop

train_dir = '圖片路徑/train'
test_dir = '圖片路徑/test'
validation_dir = '圖片路徑/validation'

model = Sequential()
#1
model.add(Conv2D(filters=64,
                 kernel_size=(3,3),
                input_shape=(250,250,3),
                 padding="same",
                 activation='relu'
                ))
#2
model.add(Conv2D(filters=64,
                 kernel_size=(3,3),
                 padding="same",
                 activation='relu'))
#3
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=128,
                 kernel_size=(3,3),
                 padding="same",
                 activation='relu'))
#4
model.add(Conv2D(filters=128,
                 kernel_size=(3,3),
                 padding="same",
                 activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

#5
model.add(Conv2D(filters=256,
                 kernel_size=(3,3),
                 padding="same",
                 activation='relu'))

#6
model.add(Conv2D(filters=256,
                 kernel_size=(3,3),
                 padding="same",
                 activation='relu'))

#7
model.add(Conv2D(filters=256,
                 kernel_size=(3,3),
                 padding="same",
                 activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

#8
model.add(Conv2D(filters=512,
                 kernel_size=(3,3),
                 padding="same",
                 activation='relu'))
#9
model.add(Conv2D(filters=512,
                 kernel_size=(3,3),
                 padding="same",
                 activation='relu'))


#10
model.add(Conv2D(filters=512,
                 kernel_size=(3,3),
                 padding="same",
                 activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

#11
model.add(Conv2D(filters=1024,
                 kernel_size=(3,3),
                 padding="same",
                 activation='relu'))

#12
model.add(Conv2D(filters=1024,
                 kernel_size=(3,3),
                 padding="same",
                 activation='relu'))

#13
model.add(Conv2D(filters=1024,
                 kernel_size=(3,3),
                 padding="same",
                 activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dropout(rate=0.2))
model.add(Dense(1024,activation='relu'))
model.add(Dense(1024,activation='relu'))
model.add(Dense(4,activation='softmax'))

model.summary()

#estop = EarlyStopping(monitor='val_loss', patience=3)
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=1e-4), metrics=['acc'] )

# 資料增強增加學習樣本
train_datagen =  ImageDataGenerator(
  rescale=1./255, #指定將影象像素縮放到0~1之間
  rotation_range=45, # 角度值，0~180，影象旋轉
  width_shift_range=0.2, # 水平平移，相對總寬度的比例
  height_shift_range=0.2, # 垂直平移，相對總高度的比例
  shear_range=0.2, # 隨機傾斜角度
  zoom_range=0.2, # 隨機縮放範圍
  horizontal_flip=True,# 一半影象水平翻轉
  fill_mode = 'nearest' #產生新的影像若有出現空白處，「以最接近的像素」填補像素
)

test_datagen = ImageDataGenerator(rescale=1./255)

# 訓練資料與測試資料  #分類超過兩類 使用categorical, 若分類只有兩類使用binary
train_generator = train_datagen.flow_from_directory(
train_dir,
target_size=(250, 250),
batch_size=20,
class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
validation_dir,
target_size=(250, 250),
batch_size=20,
class_mode='categorical',
)

# 使用批量生成器 模型模型
H = model.fit_generator(
train_generator,
steps_per_epoch=train_generator.samples/train_generator.batch_size, #一共訓練100次
epochs=15, #一共訓練回合
validation_data=validation_generator,
validation_steps=50
)

model.save('model_VGG16_food4classes_p5.h5')

# 顯示acc學習結果
accuracy = H.history['acc']
val_accuracy = H.history['val_acc']
plt.plot(range(len(accuracy)), accuracy, marker='.', label='accuracy(training data)')
plt.plot(range(len(val_accuracy)), val_accuracy, marker='.', label='val_accuracy(evaluation data)')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

# 顯示loss學習結果
loss = H.history['loss']
val_loss = H.history['val_loss']
plt.plot(range(len(loss)), loss, marker='.', label='loss(training data)')
plt.plot(range(len(val_loss)), val_loss, marker='.', label='val_loss(evaluation data)')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()