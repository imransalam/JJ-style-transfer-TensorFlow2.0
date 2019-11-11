
import os
import params
import argparse
import numpy as np
import utility_functions

import tensorflow as tf
from models import TransformerNet, get_content_loss, gram_matrix

transformer_net = TransformerNet()
transformer_optimizer = tf.keras.optimizers.Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

def get_vgg():
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    style_outputs = [vgg.get_layer(name).output for name in params.style_layers]
    content_outputs = [vgg.get_layer(name).output for name in params.content_layers]
    model_outputs = style_outputs + content_outputs
    return tf.keras.models.Model(vgg.input, model_outputs)

def compute_loss(pred_img_gram_features, pred_img_features, style_img_gram_features, content_img_content_features):
    style_score = 0
    content_score = 0

    weight_per_style_layer = 1.0 / float(params.num_style_layers)
    for init_img_gram_layer, style_img_gram_layer in zip(pred_img_gram_features, style_img_gram_features):
        style_score += weight_per_style_layer * get_content_loss(init_img_gram_layer, style_img_gram_layer)

    weight_per_content_layer = 1.0 / float(params.num_content_layers)
    for init_img_content_layer, content_img_content_layer in zip(pred_img_features, content_img_content_features):
        content_score += weight_per_content_layer * get_content_loss(init_img_content_layer, content_img_content_layer)

    loss = (style_score * params.style_weight) + (content_score * params.content_weight)
    return loss

def train_step(batch_content_images, batch_style_imgs, vgg_model):
    with tf.GradientTape() as transformer_tape:
        y_hat = transformer_net(batch_content_images)
        y_hat_vgg_features = vgg_model(y_hat)
        batch_content_images_vgg_features = vgg_model(batch_content_images)
        batch_style_images_vgg_features = vgg_model(batch_style_imgs)
        total_loss = 0
        for each_image_idx in range(len(y_hat_vgg_features)):
            pred_img_gram_features = [gram_matrix(style_layer[0]) for style_layer in y_hat_vgg_features[each_image_idx][:params.num_style_layers]]
            pred_img_content_features = [content_layer[0] for content_layer in y_hat_vgg_features[each_image_idx][params.num_style_layers:]]
            style_img_gram_features = [gram_matrix(style_layer[0]) for style_layer in batch_style_images_vgg_features[each_image_idx][:params.num_style_layers]]
            content_img_content_features = [content_layer[0] for content_layer in batch_content_images_vgg_features[each_image_idx][params.num_style_layers:]]
            total_loss = total_loss + compute_loss(pred_img_gram_features, pred_img_content_features, style_img_gram_features, content_img_content_features)


    transformer_gradients = transformer_tape.gradient(total_loss, transformer_net.trainable_variables)    
    transformer_optimizer.apply_gradients(zip(transformer_gradients, transformer_net.trainable_variables))




if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--content_img", help="Enter Content Image path here")
    parser.add_argument("--style_img", help="Enter Style Image path here")

    args = parser.parse_args()
    content_image_path = args.content_img
    style_image_path = args.style_img

    content_imgs = utility_functions.read_images(content_image_path)
    content_imgs = tf.keras.applications.vgg19.preprocess_input(content_imgs)

    style_img = utility_functions.read_image(style_image_path)
    style_img = tf.keras.applications.vgg19.preprocess_input(style_img)

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(transformer_net=transformer_net)

    epochs = 100
    batch_size = 4

    batch_style_imgs = np.repeat(style_img, batch_size, axis=0)


    vgg_model = get_vgg()
    for layer in vgg_model.layers:
        layer.trainable = False

    norm_means = utility_functions.np.array([103.939, 116.779, 123.68])
    min_vals = - norm_means
    max_vals = 255 - norm_means

    for epoch in range(epochs):
        for idx in range(0, len(content_imgs), batch_size):
            batch_content_images = content_imgs[idx: idx + batch_size]

            train_step(batch_content_images, batch_style_imgs, vgg_model)
            print("Train Step Completed")

        if (epoch + 1) % 1 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
