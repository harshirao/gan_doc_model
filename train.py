import os
import argparse
import json
import numpy as np
import tensorflow as tf
import model.data as data
import model.model as m
import model.evaluate as e
import pandas as pd
import matplotlib.pyplot as plt

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

####### Update Discriminator params
def update_disc(model, x, opt, loss, params, session):
    z = np.random.normal(0, 1, (params.batch_size, params.z_dim))
    loss, _ = session.run(
                    [loss, opt], 
                    feed_dict={
                        model.x: x,
                        model.z: z 
                    })
    return loss

####### Update Generator params
def update_gen(model, x, opt, loss, al_opt, al_loss, params, session):
    z = np.random.normal(0, 1, (params.batch_size, params.z_dim))
    loss, _, al_loss, _ = session.run(
                    [loss, opt, al_loss, al_opt], 
                    feed_dict={
                        model.x: x,
                        model.z: z 
                    })
    return loss, al_loss

####### Training
def train(model, dataset, params):
    log_dir = os.path.join(params.model, 'logs')
    model_dir = os.path.join(params.model, 'model')

    with tf.Session(config=tf.ConfigProto(
        inter_op_parallelism_threads=params.num_cores,
        intra_op_parallelism_threads=params.num_cores,
        gpu_options=tf.GPUOptions(allow_growth=True)
    )) as session:
        avg_d_loss = tf.placeholder(tf.float32, [], 'd_loss_ph')
        tf.summary.scalar('d_loss', avg_d_loss)
        avg_g_loss = tf.placeholder(tf.float32, [], 'g_loss_ph')
        tf.summary.scalar('g_loss', avg_g_loss)
        validation = tf.placeholder(tf.float32, [], 'validation_ph')
        tf.summary.scalar('validation', validation)
        avg_al_loss = tf.placeholder(tf.float32, [], 'al_loss_ph')
        tf.summary.scalar('al_loss', avg_al_loss)

        summary_writer = tf.summary.FileWriter(log_dir, session.graph)
        summaries = tf.summary.merge_all()
        saver = tf.train.Saver(tf.global_variables())

        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()

        d_losses = []
        g_losses = []
        al_g_losses = []
        df_disc_losses = pd.DataFrame()
        df_gen_losses = pd.DataFrame()
        df_losses = pd.DataFrame()
        df_val_scores = pd.DataFrame()
        df_al_losses = pd.DataFrame()

        training_data = dataset.batches('training', params.batch_size)

        best_val = 0.0
        training_labels = np.array(
            [[y] for y, _ in dataset.rows('training', num_epochs=1)]
        )
        validation_labels = np.array(
            [[y] for y, _ in dataset.rows('validation', num_epochs=1)]
        )

        for step in range(params.num_steps + 1):
            _, x = next(training_data)

            ###### update discriminator
            d_loss_step = update_disc(
                            model,
                            x,
                            model.D_solver, 
                            model.D_loss,   
                            params,
                            session
                        )
            d_losses.append(d_loss_step)
            df_disc_losses = df_disc_losses.append({'step': step, 'disc_loss': d_loss_step}, ignore_index=True)

            ###### update generator
            g_loss_list = []
            for i in range(0, params.num_gen):
                g_loss_i, al_loss = update_gen(
                                    model,
                                    x,
                                    model.G_solver[i],  
                                    model.Gen_loss[i],  
                                    model.Al_solver,  
                                    model.Al_gen_loss,  
                                    params,
                                    session)
                g_loss_list.append(g_loss_i)
            
            al_g_losses.append(al_loss)
            df_al_losses = df_al_losses.append({'step': step, 'alpha_gen_loss': al_loss}, ignore_index=True)

            df_gen_losses = df_gen_losses.append({'step': step, 
                                    'g_0_loss': g_loss_list[0], 'g_1_loss': g_loss_list[1], 
                                    'g_2_loss': g_loss_list[2], 'g_3_loss': g_loss_list[3],
                                    'g_4_loss': g_loss_list[4]
                                    }, ignore_index=True)
            g_losses.append(g_loss_list)
            
            ###### print discriminator and generators losses
            if step % params.log_every == 0:
                text = '{}: {:.6f} \t'     
                g_losses_print = g_losses
                print(text.format(
                    step,
                    d_losses[-1], 
                    ), g_losses_print[-1]   
                    , al_g_losses[-1]
                )
            
            ###### print best validation scores
            if step and (step % params.save_every) == 0:
                validation_vectors = m.vectors(
                    model,
                    dataset.batches(
                        'validation',
                        params.batch_size,
                        num_epochs=1
                    ),
                    session
                )
                training_vectors = m.vectors(
                    model,
                    dataset.batches(
                        'training',
                        params.batch_size,
                        num_epochs=1
                    ),
                    session
                )
                val = e.evaluate(
                    training_vectors,
                    validation_vectors,
                    training_labels,
                    validation_labels
                )[0]
                print('validation: {:.3f} (best: {:.3f})'.format(
                    val,
                    best_val or 0.0
                ))
                df_val_scores = df_val_scores.append({'step': step, 'val_score': val, 'best_val_score': best_val}, ignore_index=True)

                if val > best_val:
                    best_val = val
                    print('saving: {}'.format(model_dir))
                    saver.save(session, model_dir, global_step=step)
                
                summary, = session.run([summaries], feed_dict={
                    model.x: x,
                    model.z: np.random.normal(0, 1, (params.batch_size, params.z_dim)),
                    validation: val,
                    avg_d_loss: np.average(d_losses),
                    avg_al_loss: np.average(al_g_losses),
                    avg_g_loss: np.average(g_losses)
                })
                summary_writer.add_summary(summary, step)
                summary_writer.flush() 
                d_losses = []
                g_losses = []

        ###### Store and plot discriminator and generator losses
        df_losses = df_disc_losses.join(df_gen_losses.set_index('step'), on='step', how='inner')
        df_losses = df_losses.join(df_al_losses.set_index('step'), on='step', how='inner')
        df_losses.to_csv(log_dir+'_gen_disc_losses.csv', index=False)
        df_val_scores.to_csv(log_dir+'_val_scores.csv', index=False)

        img_loss = df_losses.plot(   x='step'
                                    ,y=['disc_loss', 'alpha_gen_loss'] 
                                    ,kind='line'
                                    ,title='Trainings Losses: Discriminator & Generator'
                                    ,xlabel="Step"
                                    ,ylabel="Loss").get_figure()
        img_loss.savefig(log_dir+'_disc_gen_loss.png')


def main(args):
    if not os.path.exists(args.model):
        os.mkdir(args.model)
    
    if args.d_dim == 0:
        args.d_dim = args.g_dim

    with open(os.path.join(args.model, args.param_file), 'w') as f:
        f.write(json.dumps(vars(args)))

    dataset = data.Dataset(args.dataset)
    x = tf.placeholder(tf.float32, shape=(None, args.vocab_size), name='x')
    z = tf.placeholder(tf.float32, shape=(None, args.z_dim), name='z')
    model = m.DocModel(x, z, args)
    train(model, dataset, args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help='path to model output directory')
    parser.add_argument('--dataset', type=str, required=True,
                        help='path to the input dataset')
    parser.add_argument('--num_gen', type=int, default=5,
                        help='No. of Generators to train')
    parser.add_argument('--vocab_size', type=int, default=2000,
                        help='the vocab size')
    parser.add_argument('--g_dim', type=int, default=100,
                        help='size of generator hidden dimension')
    parser.add_argument('--g_i_dim', type=int, default=256,
                        help='size of ith generator hidden dimension')
    parser.add_argument('--d_dim', type=int, default=128,
                        help='size of Discriminator hidden dimension')
    parser.add_argument('--z_dim', type=int, default=100,
                        help='size of the document encoding')
    parser.add_argument('--learning_rate', type=float, default=0.0002,
                        help='initial learning rate')
    parser.add_argument('--num_steps', type=int, default=10000,
                        help='the number of steps to train for')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='the batch size')
    parser.add_argument('--num-cores', type=int, default=2,
                        help='the number of CPU cores to use')
    parser.add_argument('--lam', type=int, default=10,
                        help='lambda value')
    parser.add_argument('--log_every', type=int, default=100,
                        help='print losses after this many steps')
    parser.add_argument('--save_every', type=int, default=500,
                        help='print validation score after this many steps')
    parser.add_argument('--validate-every', type=int, default=2000,
                        help='do validation after this many steps')
    parser.add_argument('--param_file', type=str, default='params.json',
                        help='params file name: For storing parameters used, in output-folder')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())
