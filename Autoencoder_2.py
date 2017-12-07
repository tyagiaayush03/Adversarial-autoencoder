import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
import datetime
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

#hyperparameter initialization
input_dim=784
learning_rate=0.1
n_hidden1=1000
n_hidden2=1000
n_latent=2
n_epoch=1000
result_path='./Autoencoder'
beta1=0.9
batch_size=100
#placeholder input and output
x_input=tf.placeholder(tf.float32,shape=[None,784],name='input')
y_=tf.placeholder(tf.float32,shape=[None,784],name='Output')
decoder_input=tf.placeholder(tf.float32,shape=[1,n_latent],name='Decoder_input')


def generate_image(sess,op):
    x_point=np.arange(0,1,0.25).astype(np.float32)
    y_point=np.arange(0,1,0.25).astype(np.float32)
    nx,ny=len(x_point),len(y_point)
    print(x_point,y_point,nx,ny)
    plt.subplot()
    gs = gridspec.GridSpec(nx, ny, hspace=0.05, wspace=0.05)
    for i, g in enumerate(gs):
        z = np.concatenate(([x_point[int(i / ny)]], [y_point[int(i % nx)]]))
        print('latent variable',z)
        z = np.reshape(z, (1, 2))
        x = sess.run(op, feed_dict={decoder_input: z})
        ax = plt.subplot(g)
        img = np.array(x.tolist()).reshape(28, 28)
        ax.imshow(img, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('auto')
    plt.show()

def form_results():
    folder_name="/{0}_{1}_{2}_{3}_{4}_{5}_autoencoder". \
    format(datetime.datetime.now(), n_latent, learning_rate, batch_size, n_epoch, beta1)
    tensorboard_path=result_path+folder_name+'/Tensorboard'
    saved_model_path=result_path+folder_name+'/Saved_model/'
    log_path=result_path+folder_name+'/log'
    if not os.path.exists(result_path+folder_name):
        os.mkdir(result_path+folder_name)
        os.mkdir(tensorboard_path)
        os.mkdir(saved_model_path)
        os.mkdir(log_path)
    return tensorboard_path,saved_model_path,log_path


def dense(x,n1,n2,name):
    with tf.variable_scope(name,reuse=None):
        weights=tf.get_variable("weights",shape=[n1,n2],initializer=tf.random_normal_initializer(mean=0.,stddev=0.01))
        bias=tf.get_variable("biases",shape=[n2],initializer=tf.constant_initializer(0.0))
        out=tf.add(tf.matmul(x,weights),bias)
        return out
#encoder
def encoder(x,reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variable()
    with tf.name_scope('Encoder'):
        layer1_e=tf.nn.relu(dense(x,input_dim,n_hidden1,'layer1_e'))
        layer2_e=tf.nn.relu(dense(layer1_e,n_hidden1,n_hidden2,'layer2_e'))
        latent_variable=dense(layer2_e,n_hidden2,n_latent,'latent_variable')
        return latent_variable

def decoder(x,reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Decoder'):
        layer1_d=tf.nn.relu(dense(x,n_latent,n_hidden2,'layer1_d'))
        layer2_d=tf.nn.relu(dense(layer1_d,n_hidden2,n_hidden1,'layer2_d'))
        output=tf.nn.sigmoid(dense(layer2_d,n_hidden1,input_dim,'output'))
        return output

def train(train_model):
    with tf.variable_scope(tf.get_variable_scope()):
        encoder_output=encoder(x_input)
        decoder_output=decoder(encoder_output)
    with tf.variable_scope(tf.get_variable_scope()):
        decoder_image=decoder(decoder_input,reuse=True)

    #loss
    loss=tf.reduce_mean(tf.square(y_-decoder_output))
    #optimizer
    optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1).minimize(loss)
    init=tf.global_variables_initializer()
    #visualization
    tf.summary.scalar(name='Loss',tensor=loss)
    tf.summary.histogram(name='Encoder Distribution',values=encoder_output)
    input_images=tf.reshape(x_input,[-1,28,28,1])
    generated_image=tf.reshape(decoder_output,[-1,28,28,1])
    tf.summary.image(name='Input images',tensor=input_images,max_outputs=10)
    tf.summary.image(name='generated images',tensor=generated_image,max_outputs=10)
    summary_op=tf.summary.merge_all()

    #saving model
    #training
    saver=tf.train.Saver()
    step=0
    with tf.Session() as sess:
        sess.run(init)
        if train_model:
            tensorboard_path,saved_model_path,log_path=form_results()
            writer=tf.summary.FileWriter(logdir=tensorboard_path,graph=sess.graph)
            for i in range(n_epoch):
                n_batches=int(mnist.train.num_examples/batch_size)
                for b in range(n_batches):
                    batch=mnist.train.next_batch(batch_size)
                    sess.run(optimizer,feed_dict={x_input:batch[0],y_:batch[0]})
                    if b%100==0:
                        batch_loss,summary=sess.run([loss,summary_op],feed_dict={x_input:batch[0],y_:batch[0]})
                        writer.add_summary(summary,global_step=step)
                        print("Loss:{}".format(batch_loss))
                        print("Epoch:{},iteration:{}".format(i,b))
                        with open(log_path+'log_text','a') as log:
                            log.write("Epoch: {}, iteration: {}\n".format(i, b))
                            log.write("Loss: {}\n".format(batch_loss))
                    step+=1
                saver.save(sess,save_path=saved_model_path,global_step=step)
            print("Model trained")
            print("Tensorflow path:{}".format(tensorboard_path))
            print("log Path:{}".format(log_path+'/log.txt'))
            print("Saved Model path:{}".format(saved_model_path))
        else:
            all_results=os.listdir(result_path)
            all_results.sort()
            saver.restore(sess,save_path=tf.train.latest_checkpoint(result_path +'/'+ all_results[-1]+'/Saved_model/'))
            generate_image(sess,op=decoder_image)
if __name__=='__main__':
    train(train_model=True)
