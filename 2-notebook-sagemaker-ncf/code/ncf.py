import os
import shutil
import argparse
import logging
import pandas as pd
import numpy as np
import boto3
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate, Multiply
from tensorflow.keras.optimizers import Adam

def load_dataset(data_path, logger):
    if data_path.startswith("s3://"):
        local_path = os.path.join("/tmp", "merged_data.csv")
        os.system(f"aws s3 cp {data_path} {local_path}")
        logger.info(f"Copied data from {data_path} to {local_path}")
        data_path = local_path

    return pd.read_csv(data_path)

def prepare_data(data):
    user_ids = data['user_id'].unique()
    item_ids = data['item_id'].unique()

    user_to_index = {user_id: index for index, user_id in enumerate(user_ids)}
    item_to_index = {item_id: index for index, item_id in enumerate(item_ids)}

    data['user_id'] = data['user_id'].map(user_to_index)
    data['item_id'] = data['item_id'].map(item_to_index)

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42)

    num_users = len(user_ids)
    num_items = len(item_ids)

    return train_data, val_data, test_data, num_users, num_items

def get_model(num_users, num_items, embed_dim=8):
    user_input = Input(shape=(1,), name='user_input')
    item_input = Input(shape=(1,), name='item_input')

    user_embedding = Embedding(input_dim=num_users, output_dim=embed_dim, name='user_embedding')(user_input)
    item_embedding = Embedding(input_dim=num_items, output_dim=embed_dim, name='item_embedding')(item_input)

    user_vecs = Flatten()(user_embedding)
    item_vecs = Flatten()(item_embedding)

    y = Multiply()([user_vecs, item_vecs])
    y = Dense(1, activation='sigmoid')(y)

    model = Model(inputs=[user_input, item_input], outputs=y)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['accuracy'])

    return model


def serving_input_receiver_fn():
    user_input = tf.keras.Input(dtype=tf.int32, shape=[None, 1], name="user_input")
    item_input = tf.keras.Input(dtype=tf.int32, shape=[None, 1], name="item_input")

    return tf.estimator.export.ServingInputReceiver({"user_input": user_input, "item_input": item_input}, {"user_input": user_input, "item_input": item_input})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--data_path', type=str)
    parser.add_argument("--model_dir", type=str, help="S3 path for the trained model")

    args = parser.parse_args()

    # Set default value for model_dir if not provided
    if args.model_dir is None:
        args.model_dir = os.path.join(os.path.expanduser("~"), "saved_model")

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    data = load_dataset(args.data_path, logger)

    train_data, val_data, test_data, num_users, num_items = prepare_data(data)

    model = get_model(num_users, num_items)

    model.fit(
        [train_data['user_id'].values, train_data['item_id'].values],
        train_data['interaction'].values,
        validation_data=(
            [val_data['user_id'].values, val_data['item_id'].values],
            val_data['interaction'].values
        ),
        batch_size=args.batch_size,
        epochs=args.epochs
    )

    # Save the trained model locally
    local_model_path = "saved_model/1"
    model.save(local_model_path, save_format='tf')


    # Move the saved model to /opt/ml/model
    final_model_path = "/opt/ml/model"
    shutil.move(local_model_path, final_model_path)

    # 아래는 로컬로 실행할때 사용
    #final_model_path = os.path.join(os.path.expanduser("~"), "saved_model")
    #shutil.move(local_model_path, final_model_path)

    # Upload the model to S3
    s3 = boto3.resource('s3')
    s3_bucket = args.model_dir.split('/')[2]
    s3_prefix = '/'.join(args.model_dir.split('/')[3:])

    for root, dirs, files in os.walk(local_model_path):
        for filename in files:
            local_path = os.path.join(root, filename)
            s3_path = os.path.join(s3_prefix, os.path.relpath(local_path, local_model_path))
            s3.Bucket(s3_bucket).upload_file(local_path, s3_path)

    test_loss, test_acc = model.evaluate(
        [test_data['user_id'].values, test_data['item_id'].values],
        test_data['interaction'].values
    )

    logger.info(f'Test loss: {test_loss}, Test accuracy: {test_acc}')


if __name__ == '__main__':
    main()

