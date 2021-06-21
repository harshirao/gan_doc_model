import os
import argparse
import json
from collections import namedtuple
import numpy as np
import tensorflow as tf
import model.data as data
import model.model as m
import model.evaluate as e
import pandas as pd

def evaluate(model, dataset, params):
    log_dir = os.path.join(params.model, 'logs')
    with tf.Session(config=tf.ConfigProto(
        inter_op_parallelism_threads=params.num_cores,
        intra_op_parallelism_threads=params.num_cores,
        gpu_options=tf.GPUOptions(allow_growth=True)
    )) as session:
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()

        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(params.model)
        saver.restore(session, ckpt.model_checkpoint_path)

        print('computing vectors...')

        recall_values = [0.0001, 0.0002, 0.0005, 0.002, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

        validation_labels = np.array(
            [[y] for y, _ in dataset.rows('validation', num_epochs=1)]
        )
        training_labels = np.array(
            [[y] for y, _ in dataset.rows('training', num_epochs=1)]
        )
        training_labels = np.concatenate(
            (training_labels, validation_labels),
            0
        )
        test_labels = np.array(
            [[y] for y, _ in dataset.rows('test', num_epochs=1)]
        )

        validation_vectors = m.vectors(
            model,
            dataset.batches('validation', params.batch_size, num_epochs=1),
            session
        )
        training_vectors = m.vectors(
            model,
            dataset.batches('training', params.batch_size, num_epochs=1),
            session
        )
        training_vectors = np.concatenate(
            (training_vectors, validation_vectors),
            0
        )
        test_vectors = m.vectors(
            model,
            dataset.batches('test', params.batch_size, num_epochs=1),
            session
        )

        print('evaluating...')

        results = e.evaluate(
            training_vectors,
            test_vectors,
            training_labels,
            test_labels,
            recall_values
        )

        df_precision_recall = pd.DataFrame(list(zip(recall_values, results)),
                columns=['recall','precision'])

        for i, r in enumerate(recall_values):
            print('precision @ {}: {}'.format(r, results[i]))

        ###### Plot precision-recall values
        df_precision_recall.to_csv(log_dir+'_precision_recall_values.csv', index=False)

        img_precision_recall = df_precision_recall.plot( x='recall'
                                                        ,y='precision' 
                                                        ,kind='line'
                                                        ,title='Precision vs Recall'
                                                        ,xlabel="Recall"
                                                        ,ylabel="Precision").get_figure()
        img_precision_recall.savefig(log_dir+'_precision_recall.png')


def main(args):
    with open(os.path.join(args.model, 'params.json'), 'r') as f:
        params = json.loads(f.read())
    params.update(vars(args))
    params = namedtuple('Params', params.keys())(*params.values())

    dataset = data.Dataset(args.dataset)
    x = tf.placeholder(tf.float32, shape=(None, params.vocab_size), name='x')
    z = tf.placeholder(tf.float32, shape=(None, params.z_dim), name='z')
    
    model = m.DocModel(x, z, params)
    evaluate(model, dataset, params)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help='path to model output directory')
    parser.add_argument('--dataset', type=str, required=True,
                        help='path to the input dataset')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='the batch size')
    parser.add_argument('--num-cores', type=int, default=2,
                        help='the number of CPU cores to use')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
