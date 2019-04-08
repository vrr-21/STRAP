#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[15]:


with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('cyclegan-10.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))


# In[16]:


graph = tf.get_default_graph()


# In[21]:


op_to_restore = graph.get_tensor_by_name("Model/g_B/t1:0")


# In[18]:


import cv2
ass1 = cv2.imread('assault-2.jpg')
ass2 = cv2.imread('assault-4.jpg')
demon1 = cv2.imread('demon-1.jpg')
demon2 = cv2.imread('demon-10.jpg')


# In[22]:


with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('cyclegan-34.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))
    ip1 = graph.get_tensor_by_name("input_A:0")
    ip2 = graph.get_tensor_by_name("input_B:0")

    feed_dict ={ip1:np.expand_dims(ass1, axis=0),ip2:np.expand_dims(ass2, axis=0)}
    x = sess.run(op_to_restore,feed_dict)


# In[9]:


import numpy as np


# In[13]:


from scipy.misc import imsave


# In[23]:


tensor = x
imsave('./file1.jpg',((tensor[0] + 1) * 127.5).astype(np.uint8))


# In[25]:


def cyclegan(img,img2):
    
    import tensorflow as tf
    import numpy as np
    from scipy.misc import imsave
    
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('cyclegan-10.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('./'))
    graph = tf.get_default_graph()
    op_to_restore = graph.get_tensor_by_name("Model/g_B/t1:0")
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('cyclegan-10.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('./'))
        ip1 = graph.get_tensor_by_name("input_A:0")
        ip2 = graph.get_tensor_by_name("input_B:0")

        feed_dict ={ip1:np.expand_dims(img, axis=0),ip2:np.expand_dims(img2, axis=0)}
        x = sess.run(op_to_restore,feed_dict)
        
    tensor = x
    fake_img = ((tensor[0] + 1) * 127.5).astype(np.uint8)
    imsave('./file1.jpg',((tensor[0] + 1) * 127.5).astype(np.uint8))
    
    return fake_img


    


# In[ ]:




