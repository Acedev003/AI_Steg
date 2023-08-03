import gradio as gr
import numpy  as np
import tensorflow as tf

from tensorflow.keras import layers , Model


from PIL import Image

def createmodel():
    in1 = layers.Input(shape=(None,None,3),name="in1")
    in2 = layers.Input(shape=(None,None,3),name="in2")

    concat = layers.Concatenate(name="concat1")([in1,in2])

    conv1  = layers.Conv2D( 64,(3,3),padding='same',name='conv1')(concat)
    bn1    = layers.BatchNormalization(name="bn1")(conv1)
    act1   = layers.Activation('relu',name='relu1')(bn1)

    conv2  = layers.Conv2D(128,(3,3),padding='same',name='conv2')(act1)
    bn2    = layers.BatchNormalization(name="bn2")(conv2)
    act2   = layers.Activation('relu',name='relu2')(bn2)

    conv3  = layers.Conv2D(256,(3,3),padding='same',name='conv3')(act2)
    bn3    = layers.BatchNormalization(name="bn3")(conv3)
    act3   = layers.Activation('relu',name='relu3')(bn3)

    conv4  = layers.Conv2D(512,(3,3),padding='same',name='conv4')(act3)
    bn4    = layers.BatchNormalization(name="bn4")(conv4)
    act4   = layers.Activation('relu',name='relu4')(bn4)

    #####################################################################3

    conv5  = layers.Conv2D(512,(3,3),padding='same',name='conv5')(act4)
    bn5    = layers.BatchNormalization(name="bn5")(conv5)
    act5   = layers.Activation('relu',name='relu5')(bn5)

    conv6  = layers.Conv2D(256,(3,3),padding='same',name='conv6')(act5)
    bn6    = layers.BatchNormalization(name="bn6")(conv6)
    act6   = layers.Activation('relu',name='relu6')(bn6)

    x      = layers.Concatenate(name="concat2")([act3,act6])

    conv7  = layers.Conv2D(128,(3,3),padding='same',name='conv7')(x)
    bn7    = layers.BatchNormalization(name="bn7")(conv7)
    act7   = layers.Activation('relu',name='relu7')(bn7)

    x      = layers.Concatenate(name="concat3")([act2,act7])

    conv8  = layers.Conv2D( 64,(3,3),padding='same',name='conv8')(x)
    bn8    = layers.BatchNormalization(name="bn8")(conv8)
    act8   = layers.Activation('relu',name='relu8')(bn8)

    x      = layers.Concatenate(name="concat4")([act1,act8])

    conv9  = layers.Conv2D( 3,(3,3),padding='same',name='conv9')(x)
    bn9    = layers.BatchNormalization(name="bn9")(conv9)
    act9   = layers.Activation('sigmoid',name='out1')(bn9)

    out1   = act9

    #-----------------------------------------------------------#

    rconv1  = layers.Conv2D( 64,(3,3),padding='same',name='rconv1')(out1)
    rbn1    = layers.BatchNormalization(name="rbn1")(rconv1)
    ract1   = layers.Activation('relu',name='rrelu1')(rbn1)

    rconv2  = layers.Conv2D( 128,(3,3),padding='same',name='rconv2')(ract1)
    rbn2    = layers.BatchNormalization(name="rbn2")(rconv2)
    ract2   = layers.Activation('relu',name='rrelu2')(rbn2)

    rconv3  = layers.Conv2D( 256,(3,3),padding='same',name='rconv3')(ract2)
    rbn3    = layers.BatchNormalization(name="rbn3")(rconv3)
    ract3   = layers.Activation('relu',name='rrelu3')(rbn3)

    rconv4  = layers.Conv2D( 128,(3,3),padding='same',name='rconv4')(ract3)
    rbn4    = layers.BatchNormalization(name="rbn4")(rconv4)
    ract4   = layers.Activation('relu',name='rrelu4')(rbn4)

    rconv5  = layers.Conv2D( 64,(3,3),padding='same',name='rconv5')(ract4)
    rbn5    = layers.BatchNormalization(name="rbn5")(rconv5)
    ract5   = layers.Activation('relu',name='rrelu5')(rbn5)

    rconv6  = layers.Conv2D( 3,(3,3),padding='same',name='rconv6')(ract5)
    rbn6    = layers.BatchNormalization(name="rbn6")(rconv6)
    ract6   = layers.Activation('sigmoid',name='out2')(rbn6)

    out2    = ract6

    model  = Model([in1,in2],[out1,out2])

    return model
        
model = createmodel()    

INITIAL_LR = 0.001
BETA       = 0.75

adam = tf.keras.optimizers.Adam(
    learning_rate=INITIAL_LR,
)

def stegloss(true,pred):
    l1 = true[0] - pred[0]
    l1 = tf.math.abs(l1)
    
    l2 = true[1] - pred[1]
    l2 = tf.math.abs(l2)
    
    return l1 + BETA*l2

model.compile(
    optimizer = adam,
    loss      = stegloss,
    metrics   = []
)

model.load_weights('./checkpoints')


def predict_model(cover,steg):
    cover = tf.image.convert_image_dtype(cover,tf.float32)
    cover = tf.expand_dims(cover,axis=0)
    
    steg = tf.image.convert_image_dtype(steg,tf.float32)
    steg = tf.image.resize(steg,cover.shape[1:-1])
    steg = tf.expand_dims(steg,axis=0)
    
    preds = model.predict([cover,steg])
    
    print(preds[0].shape)
    
    o1 = Image.fromarray((np.array(preds[0][0]) * 255).astype(np.uint8))
    o2 = Image.fromarray((np.array(preds[1][0]) * 255).astype(np.uint8))
    
    return (o1,o2)

inputs  = [gr.Image(label='Cover'),gr.Image(label='Steg')]
outputs = [gr.Image(label='Cover Generated'),gr.Image(label='Steg Decoded')]

app = gr.Interface(fn=predict_model, inputs=inputs, outputs=outputs,examples=[['./assets/examples/creature.jpg',"./assets/examples/butterfly.jpeg"],
                                                                              ["./assets/examples/butterfly.jpeg","./assets/examples/bw.jpg"],
                                                                              ["./assets/examples/bw.jpg","./assets/examples/dark.jpeg"],
                                                                              ["./assets/examples/dark.jpeg","./assets/examples/forestfire.JPG"],
                                                                              ["./assets/examples/forestfire.JPG","./assets/examples/heli.JPG"],
                                                                              ["./assets/examples/heli.JPG","./assets/examples/hillbefore.jpg"],
                                                                              ["./assets/examples/hillbefore.jpg",'./assets/examples/creature.jpg']])
app.launch()




