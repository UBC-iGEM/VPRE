"""
Created on Mon Aug 03 09:37:18 2020

@author: UBC iGEM 2020

Referenced Flu Forcaster constructed by Eric Ma
"""

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from random import sample
from datetime import datetime
from numpy import savetxt
from collections import Counter
from copy import deepcopy

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import sklearn
from sklearn.preprocessing import LabelBinarizer
from keras.models import model_from_yaml
import tensorflow as tf
from keras.layers import Input, Dense, Lambda, Dropout #LSTM, RepeatVector
from keras.models import Model, model_from_json
from keras import backend as K
from keras import objectives
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

import pymc3 as pm
from pymc3 import gp
import theano.tensor as tt


def load_sequence_and_metadata(kind="csv"):
    """
    Returns the sequences as a list of SeqRecords, and metadata as a pandas
    DataFrame.
    """
    starttime = datetime.now()
    sequences = [
        s for s in SeqIO.parse("Dataset/FinalSeqs.fasta", "fasta")
    ]
    for s in sequences:
      index = s.id.split("|")[0]
      index = index.split(">")[0]
      s.id = index

    if kind == "csv":
        metadata = pd.read_csv(
            "Dataset/FinalIndex.csv",
            parse_dates=["Collection_Date"],
        )

    endtime = datetime.now()
    elapsed = endtime - starttime
    print(f"load_sequence_and_metadata() took {elapsed} seconds.")
    return sequences, metadata


def load_prediction_coordinates():
    """
    Returns the coordinates of the predictions, and their associated colours,
    as a pandas DataFrame.
    """
    logger.debug("started load_prediction_coordinates()")
    df = pd.read_csv(
        "oneQ_prediction_coords_with_colors.csv", index_col=0
    )
    logger.debug("finished load_prediction_coordinates()")
    return df




def right_pad(sequences):
    """
    Pads sequences with extra "*" characters.
    """
    padded_sequences = deepcopy(sequences)
    seq_lengths = compute_seq_lengths(sequences)

    for s in padded_sequences:
        while len(s) < max(seq_lengths.keys()):
            s.seq += '*'
    return padded_sequences


def compute_seq_lengths(sequences):
    """
    Computes the sequence lengths.
    """
    seq_lengths = [len(s) for s in sequences]
    seq_lengths = Counter(seq_lengths)
    return seq_lengths


def seq2chararray(sequences):
    """
    Returns sequences coded as a numpy array. Doesn't perform one-hot-encoding.
    """
    padded_sequences = right_pad(sequences)
    seq_lengths = compute_seq_lengths(sequences)
    char_array = np.chararray(shape=(len(sequences), max(seq_lengths.keys())),
                              unicode=True)
    for i, seq in enumerate(padded_sequences):
        char_array[i, :] = list(seq)
    return char_array


def compute_alphabet(sequences):
    """
    Returns the alphabet used in a set of sequences.
    """
    alphabet = set()
    for s in sequences:
        alphabet = alphabet.union(set(s))
    alphabet.add("B")
    alphabet.add("Z")
    alphabet.add("*")
    alphabet.add("-")

    return alphabet
def encode_array(sequences):
    """
    Performs binary encoding of the sequence array.
    Inputs:
    =======
    - seq_array: (numpy array) of characters.
    - seq_lengths: (Counter) dictionary; key::sequence length; value::number of
                   sequences with that length.
    """
    # Binarize the features to one-of-K encoding.
    alphabet = compute_alphabet(sequences)
    seq_lengths = compute_seq_lengths(sequences)
    seq_array = seq2chararray(sequences)
    lb = LabelBinarizer()
    lb.fit(list(alphabet))
    print(len(alphabet))

    encoded_array = np.zeros(shape=(seq_array.shape[0],
                                    max(seq_lengths.keys()) * len(alphabet)))

    for i in range(seq_array.shape[1]):
        encoded_array[:, i*len(alphabet):(i+1)*len(alphabet)] = \
            lb.transform(seq_array[:, i])

    return encoded_array


def embedding2binary(decoder, predictions):
    """
    Decodes the predictions into a binary array.
    Inputs:
    =======
    - decoder: a Keras model.
    - predictions: a numpy array corresponding to the lower dimensional
                   projection.
    Returns:
    ========
    - a binary encoding numpy array that corresponds to a predicted sequence.
    """
    return np.rint(decoder.predict(predictions))


def binary2chararray(sequences, binary_array):
    """
    Converts a binary array into a character array.
    """

    alphabet = compute_alphabet(sequences)
    seq_lengths = compute_seq_lengths(sequences)
    seq_array = seq2chararray(sequences)

    lb = LabelBinarizer()
    lb.fit(list(alphabet))

    char_array = np.chararray(shape=(len(binary_array),
                              max(seq_lengths.keys())), unicode=True)

    for i in range(seq_array.shape[1]):
        char_array[:, i] = lb.inverse_transform(
            binary_array[:, i*len(alphabet):(i+1)*len(alphabet)])

    return char_array


def save_model(model, path):
    with open(path + '.yaml', 'w+') as f:
        model_yaml = model.to_yaml()
        f.write(model_yaml)

    model.save_weights(path + '.h5')
def load_model(path):
    with open(path + '.yaml', 'r+') as f:
        yaml_rep = ''
        for l in f.readlines():
            yaml_rep += l

    model = model_from_yaml(yaml_rep)
    model.load_weights(path + '.h5')

    return model


def get_density_interval(percentage, array, axis=0):
    """
    Returns the highest density interval on the array.
    Parameters:
    ===========
    percentage: (float, int) value between 0 and 100, inclusive.
    array: a numpy array of numbers.
    """
    low = (100 - percentage) / 2
    high = (100 - low)

    lowp, highp = np.percentile(array, [low, high], axis=axis)

    return lowp, highp


##############################################################################################
# VAE Training
##############################################################################################

# Loading dataset
simulated_seq = list(SeqIO.parse("Dataset/ranTS_full.fasta", "fasta"))
split_len = int(0.8*len(simulated_seq))
training_idxs = [i for i, s in enumerate(simulated_seq[:split_len])]
test_idxs = [i for i, s in enumerate(simulated_seq[split_len:])]

sequence_array = encode_array(simulated_seq)
training_array = sequence_array[training_idxs]
test_array = sequence_array[test_idxs]

# Setting up training
with tf.device('/gpu:0'):
    intermediate_dim = 1000
    encoding_dim = 3 # the 3 digits at the connecting point between encoder and decoder
    latent_dim = encoding_dim
    epsilon_std = 1.0
    nb_epoch = 50

    x = Input(shape=(training_array.shape[1],))
    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(latent_dim, ), mean=0.,
                                  stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon
    def vae_loss(x, x_decoded_mean):
        xent_loss = training_array.shape[1] * objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return xent_loss + kl_loss

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    x_decoded_mean = Dense(training_array.shape[1], activation='sigmoid')(z_mean)

    vae = Model(x, x_decoded_mean)
    vae.compile(optimizer='adam', loss=vae_loss)

    # build a model to project inputs on the latent space
    encoder = Model(x, z_mean)
    encoder_var = Model(x, z_log_var)

    x_train, x_test = train_test_split(training_array)

    early_stopping = EarlyStopping(monitor="val_loss", patience=2)


    # build the decoder
    encoded_input = Input(shape=(encoding_dim,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = vae.layers[-1]
    # create the decoder model
    decoder = Model(inputs=encoded_input, outputs=decoder_layer(encoded_input))
    # Train the VAE to learn weights
    h = vae.fit(x_train, x_train,
            shuffle=True,
            epochs=nb_epoch,
            validation_data=(x_test, x_test),
           )

    # Plotting training progression 
    # df = pd.DataFrame()
    # df['loss'] = h.history['loss']
    # df['val_loss'] = h.history['val_loss']
    # df.to_csv('training_data.csv',index=False)
    # plt.figure(figsize = [8,6]);
    # plt.plot(h.history['loss'],'r', linewidth=2.0);
    # plt.plot(h.history['val_loss'],'b', linewidth=2.0);
    # plt.legend(['Training loss', 'Validation Loss'], fontsize=18);
    # plt.xlabel('Epochs ', fontsize=16)
    # plt.ylabel('Loss', fontsize=16)
    # plt.title('Loss Curves', fontsize=16)
    # plt.savefig('results/history.pdf',bbox_inches='tight')
    save_model(vae, 'final_models/vae')
    save_model(encoder, 'final_models/encoder')
    save_model(decoder, 'final_models/decoder')
##############################################################################################
# GP
##############################################################################################

# Loading the protein sequence FASTA file and metadata.
sequences, metadata = load_sequence_and_metadata(kind='csv')

# Filter for just human sequences, then split into training (date < 5/30/2020) and test set (date > 5/30/2020)
training_metadata = metadata[metadata['Collection_Date'] < datetime(2020, 5, 30)]
training_idxs = [i for i, s in enumerate(sequences) if str(s.id) in training_metadata['Accession'].values]

test_metadata = metadata[metadata['Collection_Date'] >= datetime(2020, 5, 30)]
test_idxs = [i for i, s in enumerate(sequences) if str(s.id) in test_metadata['Accession'].values]

# Encode as array
sequence_array = encode_array(sequences) # encode_array is a function defined above for binary encoding
training_array = sequence_array[training_idxs]
test_array = sequence_array[test_idxs]

training_sequences = [sequences[i] for i in training_idxs]
test_sequences = [sequences[i] for i in test_idxs]

with open('final_models/vae.yaml', 'r+') as f:
    yaml_spec = f.read()

# Loading the trained models
vae = load_model('final_models/vae')
encoder = load_model('final_models/encoder')
decoder = load_model('final_models/decoder')

# Calling encoder and get the 3-digit encoded arrays for the training set from encoder
training_embeddings_mean = encoder.predict(training_array)
test_embeddings_mean = encoder.predict(test_array)

training_metadata.loc[:, 'coords0'] = training_embeddings_mean[:, 0]
training_metadata.loc[:, 'coords1'] = training_embeddings_mean[:, 1]
training_metadata.loc[:, 'coords2'] = training_embeddings_mean[:, 2]

tm_coords = deepcopy(training_metadata)  # tm_coords means "training metadata with coordinates"
tm_coords['coord0'] = training_embeddings_mean[:, 0]
tm_coords['coord1'] = training_embeddings_mean[:, 1]
tm_coords['coord2'] = training_embeddings_mean[:, 2]

tm_coords = tm_coords.reset_index()
avg_coords_by_day = tm_coords[['Collection_Date','coord0', 'coord1', 'coord2']].dropna().reset_index(drop=True) #not taking average seq

one = avg_coords_by_day.copy()
one['Collection_Date'] =pd.to_datetime(one.Collection_Date)
one['Days'] = None
first_date = one['Collection_Date'][0]
one['Days'] = ((one['Collection_Date'] - first_date)/ np.timedelta64(1, 'D')).astype(int)
one = one.sort_values(by='Collection_Date',ascending=True)

one.to_csv('Coordinate_data_new.csv',index=False)


# Step 4 Building a Gaussian process model
x_vals = np.array(one['Days']).reshape(-1, 1).astype('float32')

def build_coords_model(coordinate):
    y_vals = one[coordinate].values.astype('float32')

    print(x_vals.shape, y_vals.shape)

    with pm.Model() as model:
        l = pm.Uniform('l', 0, 30)                            # Continuous uniform log-likelihood (f = 1/(30-0) for x in [0,30], otherwise f = 0)

        # Covariance function
        log_s2_f = pm.Uniform('log_s2_f', lower=-10, upper=5) #  (f = 1/(5-(-10)) for x in [-10,5], otherwise f = 0)
        exp_component = tt.exp(log_s2_f)                      #   Returns a variable representing the exponential of a, ie e^a.
        s2_f = pm.Deterministic('s2_f', exp_component)        #   Basically wrap the function in exp_component in the form that is compatible for pm to use
        f_cov = s2_f * pm.gp.cov.ExpQuad(input_dim=1, ls=l)   # Exponentiated Quadratic

        # Sigma
        log_s2_n = pm.Uniform('log_s2_n', lower=-10, upper=5)
        s2_n = pm.HalfCauchy('s2_n', beta=2)                  # Continuous Half-Cauchy log-likelihood.(high around 0 and converge to 0 as x goes to beta)

        # GP
        gp_f = pm.gp.Latent(cov_func=f_cov)
        f = gp_f.prior("f", X=x_vals)
        σ = pm.HalfCauchy("σ", beta=5)
        ν = pm.Gamma("ν", alpha=2, beta=0.1) 
        y_ = pm.StudentT("y", mu=f, lam=1.0 / σ, nu=ν, observed=y_vals)
        ###############################################

        trace = pm.sample(draws=2000)

        pp_x = np.arange(one['Days'].tolist()[0],one['Days'].tolist()[-1]+30,1)[:, None]

    with model:
        f_pred = gp_f.conditional("f_pred", pp_x)
        gp_samples = pm.sample_posterior_predictive(trace, vars=[f_pred], samples=1000)

    return gp_samples

# Perform Gaussian regression on each dimension
coord0_preds = build_coords_model('coord0')

coord1_preds = build_coords_model('coord1')

coord2_preds = build_coords_model('coord2')

savetxt('coord0_preds.csv', coord0_preds['f_pred'], delimiter=',')
savetxt('coord1_preds.csv', coord1_preds['f_pred'], delimiter=',')
savetxt('coord2_preds.csv', coord2_preds['f_pred'], delimiter=',')

def plot_coords_with_groundtruth(coord_preds, data, ax):
    # pp_x = np.arange(len(avg_coords_by_day)+2)[:, None]
    pp_x = np.arange(one['Days'].tolist()[0],one['Days'].tolist()[-1]+30,1)[:, None]
    # pp_x = np.arange(0,len(coord_preds),1)[:,None]
    # preds =  coord_preds['f_pred']
    for x in coord_preds:
      ax.plot(pp_x, x, color='#a37eba', alpha=0.1,linewidth=0.7)
    ax.plot([], [], color='#a37eba', alpha=1,linewidth=0.7,label='Regression') 
    ax.scatter(x_vals, data, color='#68b0ab',s=10,zorder=10,label='Observation');
    ax.set_xlabel("Days (since 2020-02-21)");
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return ax

# Plotting GP results
fig = plt.figure(figsize=(15,5))
ax0 = fig.add_subplot(1,3,1)
ax1 = fig.add_subplot(1,3,2)
ax2 = fig.add_subplot(1,3,3)
ax0.set_title("Coordinate 0")
ax0.set_ylabel('Coordinate value')
ax0.set_ylim(-7.5,-2.5)
ax1.set_title("Coordinate 1")
ax1.set_ylim(-5,0.5)
ax2.set_title("Coordinate 2")
ax2.set_ylim(1,6)
plot_coords_with_groundtruth(coord0_preds['f_pred'], one['coord0'], ax0)
plot_coords_with_groundtruth(coord1_preds['f_pred'], one['coord1'], ax1)
plot_coords_with_groundtruth(coord2_preds['f_pred'], one['coord2'], ax2)
ax2.set_xlim(0,150)
ax1.set_xlim(0,150)
ax0.set_xlim(0,150)
# fig.savefig("results/GP_visualize.pdf",bbox_inches='tight',transparent=True)

# Plotting the distribution of predictions
fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(1,3,1)
ax1.hist(one_month_pred[:,0], color = '#C470E5', edgecolor = 'none',
         bins = int(180))
ax1.set_xlabel('predicted value day 135 since '+  str(first_date)[:10])
ax1.set_ylabel('Frequency')
ax1.set_title('Coordinate 0')

ax2 = fig.add_subplot(1,3,2)
ax2.hist(one_month_pred[:,1], color = '#C470E5', edgecolor = 'none',
         bins = int(180))
ax2.set_xlabel('predicted value at day 135 since '+  str(first_date)[:10])
ax2.set_title('Coordinate 1')

ax3 = fig.add_subplot(1,3,3)
ax3.hist(one_month_pred[:,2], color = '#C470E5', edgecolor = 'none',
         bins = int(180))
ax3.set_xlabel('predicted value at day 135 since '+  str(first_date)[:10])
ax3.set_title('Coordinate 2')
# fig.savefig('results/pre2.pdf',bbox_length='tight')


# Testing against testing sequences
test_coords_embed = deepcopy(test_metadata)
test_coords_embed['coord0'] = test_embeddings_mean[:, 0]
test_coords_embed['coord1'] = test_embeddings_mean[:, 1]
test_coords_embed['coord2'] = test_embeddings_mean[:, 2]
test_coords_embed = test_coords_embed[~test_coords_embed['Accession'].isna()]

decoder.predict(test_coords_embed[['coord0', 'coord1', 'coord2']].values)

one_month_pred = np.array([coord0_preds['f_pred'][:, -1], coord1_preds['f_pred'][:, -1], coord2_preds['f_pred'][:, -1]]).T
savetxt('one_month_pred.csv', one_month_pred, delimiter=',')

def embedding2seqs(decoder, preds, sequences):
    binary = embedding2binary(decoder, preds)
    chararray = binary2chararray(sequences, binary)
    strs = [''.join(i for i in k) for k in chararray]
    strcounts = Counter(strs)
    seqrecords = []
    for s, c in sorted(strcounts.items(), key=lambda x: x[1], reverse=True):
        s = Seq(s)
        sr = SeqRecord(s, id='weight_' + str(c))
        seqrecords.append(sr)

    return binary, chararray, strcounts, strs, seqrecords

oneQ_binary, oneQ_chararray, oneQ_strcounts, oneQ_strs, oneQ_seqrecords = embedding2seqs(decoder, one_month_pred, sequences)
sorted(oneQ_strcounts.values(), reverse=True)[0:3]
savetxt('one_mon_pred_seq.csv',oneQ_strcounts,delimiter=',')

most_probable_seq = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQGVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT"
levDs_from_preds = [distance(str(record.seq), most_probable_seq) for record in test_sequences]

# Plotting comparison with testing/training sequences
comparing_data = 'training'
def ecdf_scatter(data,ax):
    x, y = np.sort(data), np.arange(1, len(data)+1) / len(data)
    ax.scatter(x, y,color='#679b9b')
    ax.set_title('Comparing with %s sequences' %comparing_data)
    ax.set_xlabel('Number of amino acids differing')
    ax.set_ylabel('Cumulative distribution')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig('results/comparison_with_%s_seq.pdf' %comparing_data,transparent=True)
    
fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot()
ecdf_scatter(levDs_from_preds,ax)
