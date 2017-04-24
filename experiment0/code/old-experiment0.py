#!/usr/bin/env python3

# precondition: must be running NVIDIA digits server on localhost. 
# script requires a unix-like environment

# TO DO: MORE ERROR CHECKING. REFACTOR INTO A NEAT CLASS/QUASI NVIDIA-DIGITS API

"""
experiment0 consists of the following:

    1. taking a seismic (CMP stacked) image
    2. dividing it up into 2k+1 by 2k+1 windows
    3. some of the windows for we which we have a class label
        ('event' or 'no event' depending on the classification of the central pixel)
        will be part of the training/validation set.
    4. all the other windows will be part of the test set.
    5. we'll send POST requests to the NVIDIA DIGITS server at localhost in order to:
        a) create a dataset
        b) create a model
        c) classify the test set, stashing the results
    6. since we don't necessarily have class labels for the test set,
        we will simply generate an image with a color map that reflects the probability of event/noevent
    7. we display the image
"""

import argparse
import os
import json
import re

from gs_utilities import run_command
from gs_utilities import print_command_info

import pymei as pm
import numpy as np
import matplotlib.pyplot as plt

# process command line options
parser = argparse.ArgumentParser(description=globals()['__doc__']) # the docstring above
parser.add_argument('-w', '--window-size', type=int, default=33, help="define the (odd) size, 2K+1 by 2K+1, of each cutout of the main image")
parser.add_argument('-n', '--name', default='', help='give this dataset/model/classification a name')
parser.add_argument('-l', '--log', default='', help="log all console output to this file")
parser.add_argument('-v', '--verbose', action='store_true', help='print diagnostic info')
parser.add_argument('-c', '--cookie-file', default='digits.cookie', help='cookie file for NVIDIA DIGITS server (used for login)')

parser.add_argument('-t', '--training-folder', default='', help='path to folder where the training images are located (split into subfolders whose names are the class labels). if not provided, the training images will be generated')
parser.add_argument('-cd', '--create-dataset', action='store_true', help='create a dataset for training')
parser.add_argument('-d', '--dataset', default='', help="specify dataset to use.")
parser.add_argument('-cm', '--create-model', action='store_true', help='create a model for training')
parser.add_argument('-m', '--model', default='', help="specify model to use.")
parser.add_argument('-tst', '--test-folder', default='', help='path to folder where the test images are. if not provided, test images will be generated')
parser.add_argument('-cl', '--classify', action='store_true', help='classify the images in the test folder, generating class probabilities for each')

parser.add_argument('image', help='SEGY image on which to perform experiment0')
args = parser.parse_args()





# if window size isn't odd, there is no central pixel
if args.window_size < 0 or args.window_size % 2 != 1:
	raise ArgumentError("must specify a valid odd window size argument")
half_window_size = args.window_size // 2


# name of this run of the experiment, followed by the timestamp to put on files/names we generate
timestamp = + args.name + '--' + run_command('date "+%Y-%m-%d--%H-%M-%S"')['stdout'].replace('\n','')


# read the seismic image:

# pm.load returns a SEGY object (containing data from the SEGY file - the textual header (3200 byte EBCDIC), the binary header (400 byte), and the header+payload for each seismic trace (header is 240 bytes, payload is 4*ns bytes (there are ns samples in a trace, each one being a float (possibly an 'IBM float'))))
# then segy_traces will be a list of Trace objects corresponding to each seismic trace in the file (header and payload)
segy_traces = [trace for trace in pm.load(args.image)]

img = np.array([trace.data for trace in segy_traces]).T # array of samples with each cdp. accessed in (time, cdp) order.
vmax = np.max(np.abs(img)) # get the maximum value over all the points (to use for the color map of the image)
#vmin = np.min(np.abs(img))
vmin = -vmax 




# see https://github.com/NVIDIA/DIGITS/blob/be2bc827cab5a7bc3f9c2e5baa9d8cd48fd414eb/docs/API.md
# for some examples of the digits API ('docs' folder, master branch of the nvidia digits repository)


# check that we have a cookie to login to the nvidia digits server
if run_command('ls ' + args.cookie_file)['returncode'] != 0: # login cookie not found, so bake it:
    username = run_command('whoami')['stdout']
    bake_cookie_command =('curl localhost/login' +
                          ' -c ' + args.cookie_file + 
                          ' -XPOST' +
                          ' -F username=' + username)
    bk = run_command(bake_cookie_command)
    if args.verbose:
        print_command_info(bk) # print diagnostic information






# generate training images, if necessary
if args.training_folder != '':
    training_folder = args.training_folder
else:
    training_folder = 'train--' + timestamp 
    run_command('mkdir -p ' + training_folder)
    run_command('mkdir -p ' + training_folder + '/noevent')
    run_command('mkdir -p ' + training_folder + '/event')


    # Data with 'non events' followed by labeled seismic events:
    for example_points, class_label in [(np.loadtxt('noevent-points.txt'), 'noevent'), (np.loadtxt('event-points.txt'), 'event')]:
        # each text file has (cdp, time) pairs for each labeled point
        training_points = {int(cdp): time for cdp, time in example_points} # this dict maps cdp to time coordinate

        for trace in sgy_traces:
            if trace.cdp in training_points:
                training_points[trace.cdp] = int(training_points[trace.cdp] / 1e3 / trace.dt) # convert the time (in ms) found in the file to a sample (vertical axis) number

        for i, pt in enumerate(training_points):
            cdp_coord = pt
            time_coord = training_points[pt]

            # warning issued if any point is too close to the border. validity of the model of 'central pixel' called into question
            if (time_coord-half_window_size<0 or 
                    time_coord+half_window_size>=img.shape[0] or 
                    cdp_coord-half_window_size<0 or 
                    cdp_coord+half_window_size>=img.shape[1]):
                if args.verbose:
                    print('WARNING: the window for point {} overflows image bounds' % i)

            tmin = max(0, time_coord-half_window_size)
            tmax = min(img.shape[0], time_coord+half_window_size+1)
            cdpmin = max(0, cdp_coord-half_window_size)
            cdpmax = min(img.shape[1], cdp_coord+half_window_size+1)

            window = img[tmin:tmax, cdpmin:cdpmax]

            plt.imsave('{}/{}/image{}-time{}-cdp{}.png'.format(training_folder, class_label, i, time_coord, cdp_coord), window, cmap='gray', vmin=vmin, vmax=vmax)

# training images have been generated 





if args.create_dataset: 
    # create a new dataset
    create_dataset_command =('curl localhost/datasets/images/classification.json' +
                             ' -b ' + args.cookie_file + # must login to create a dataset
                             ' -XPOST' + # use POST method
                             ' -F folder_train=' + os.getcwd() + '/' + training_folder + # contains subfolders whose names are the class labels for the training images therein
                             ' -F encoding=png' +
                             ' -F resize_channels=1' + # greyscale image: 1 color channel
                             ' -F resize_height=' + str(args.window_size) + 
                             ' -F resize_width=' + str(args.window_size) + 
                             ' -F method=folder' +
                             ' -F dataset_name=experiment0--' + timestamp)
    create_dataset = run_command(create_dataset_command)
    if args.verbose:
        print_command_info(create_dataset) #print(json.dumps(create_dataset, indent=4))
    dataset_id = json.loads(create_dataset['stdout'])['id'] # we'll use this id to refer to the dataset in future operations

    # wait until dataset is ready
    waiting = True
    while waiting:
        run_command('sleep 1')
        q = run_command('curl localhost/datasets/' + dataset_id + '/status')
        if json.loads(q['stdout'])['status'] == 'Done':
            if args.verbose:
                print_command_info(q)
            waiting = False # dataset has been successfully created
    
    if args.verbose:
        q = run_command('curl localhost/datasets/' + dataset_id + '.json') # query more information about the dataset
        print_command_info(q)

    # return some metadata about datasets and classification models stored in the DB
    if args.verbose:
        q = run_command('curl localhost/index.json')
        print_command_info(q)

elif args.dataset != '': # use existing dataset (must fetch its id)
    q = run_command('curl localhost/index.json')
    print_command_info(q)
    for m in json.loads(q['stdout'])['datasets']:
        if m['name'] == args.name:
            dataset_id = m['id']







if args.create_model:
    # create a new classification model
    create_model_command =('curl localhost/models/images/classification.json' +
                           ' -b ' + args.cookie_file +
                           ' -XPOST' +
                           ' -F method=standard' + # use one of the standard networks: LeNet, AlexNet, GoogLeNet
                           ' -F standard_networks=lenet' + # the more basic, time-honored convolutional neural network
                           ' -F framework=caffe' +
                           ' -F train_epochs=30' + # number of passes through the training data
                           ' -F solver_type=SGD' + # stochastic gradient descent
                           ' -F learning_rate=0.01' + # step size during gradient descent
                           ' -F use_mean=image' + # subtracts mean image from the data
                           # ' - F random_seed=42' # set a seed to reproduce training results for multiple runs
                           ' -F dataset=' + dataset_id +
                           ' -F model_name=experiment0--' + timestamp)
    create_model = run_command(create_model_command)
    if args.verbose:
        print_command_info(create_model)
    model_id = json.loads(create_model['stdout'])['id']

    # wait until model is ready
    waiting = True
    while waiting:
        if args.verbose:
            print('running model. please stay calm.')
        run_command('sleep 10')
        q = run_command('curl localhost/models/' + model_id + '.json')
        if json.loads(q['stdout'])['status'] == 'Done':
            if args.verbose:
                print_command_info(q)
            waiting = False # dataset has been successfully created

    # return some metadata about datasets and classification models stored in the DB
    if args.verbose:
        q = run_command('curl localhost/index.json')
        print_command_info(q)

elif args.model != '': # use existing model (must fetch its id)
    q = run_command('curl localhost/index.json')
    print_command_info(q)
    for m in json.loads(q['stdout'])['models']:
        if m['name'] == args.name:
            model_id = m['id']





# generate the test images, if necessary
if args.test_folder != '':
    test_folder = args.test_folder
else:
    test_folder = 'test--' + timestamp
    run_command('mkdir -p ' + test_folder)

    for cdp_index in range(half_window_size, img.shape[1]-half_window_size):
        # one separate test image folder for each cdp (seismic trace/time series)
        test_image_folder = '{}/cdp{}'.format(test_folder, cdp_index)
        run_command('mkdir -p ' + test_image_folder)
        image_paths_file = '{}/image_paths.txt'.format(test_image_folder) # file with the path of test images to be classified
        
        with open(image_paths_file, 'w') as listofimagepaths:
            for time_index in range(half_window_size, img.shape[0]-half_window_size):
                
                window = img[time_index - half_window_size  :  time_index + half_window_size+1, cdp_index - half_window_size  :  cdp_index + half_window_size+1]
                
                image_filename = '{}/time{}-cdp{}.png'.format(test_image_folder, time_index, cdp_index)
                plt.imsave(image_filename, window, cmap='gray', vmin=vmin, vmax=vmax)
                listofimagepaths.write(image_filename + '\n')












# now classify each batch of test images we generated:

if args.classify:
    for cdp_index in range(half_window_size, img.shape[1]-half_window_size):
        # one separate test image folder for each cdp (seismic trace/time series)
        test_image_folder = '{}/cdp{}'.format(test_folder, cdp_index)
        image_paths_file = '{}/image_paths.txt'.format(test_image_folder) # file with the path of test images to be classified
        predictions_file = '{}/predictions'.format(test_image_folder)
        classify_many_command =('curl localhost/models/images/classification/classify_many.json' +
                                ' -XPOST' +
                                ' -F job_id=' + model_id +
                                ' -F image_folder=' + os.getcwd() + # prepend this path to the image paths in the list
                                ' -F image_list=@' + image_paths_file + # each line has the path for a test image
                                ' > ' + predictions_file)
        c = run_command(classify_many_command)
        if args.verbose:
            print('commandline:', c['commandline'], '\nreturncode:', c['returncode'], '\nstderr:', c['stderr'], sep='\n')









# now prepare the image for display with a simple color map from 0 to 100 % chance of seismic event at each pixel, as predicted above

event_probabilities = np.zeros(img.shape)
time_index_pattern = re.compile(r'time([0-9]+)-cdp[0-9]+\.png$') # extract the time index from the image path

for cdp_index in range(half_window_size, img.shape[1]-half_window_size):
    # one separate test image folder for each cdp (seismic trace/time series)
    test_image_folder = '{}/cdp{}'.format(test_folder, cdp_index)
    predictions_file = '{}/predictions'.format(test_image_folder)
        
    with open(predictions_file, 'r') as pf:
        predictions = pf.read()

    # predictions is a dictionary of (image_path, array of [label, probability] arrays, one for each class in order of most probable) mappings
    predictions = json.loads(predictions)['classifications']
    for image_path in predictions:
        time_index = int(time_index_pattern.search(image_path).group(1)) # extract the time sample number from the image path
        
        # extract the probability (in %) of 'event' (as opposed to the complementary 'noevent')
        # the most likely of the two appears first in the classification array
        if predictions[image_path][0][0] == 'event':
            prob_event = predictions[image_path][0][1]
        else:
            prob_event = predictions[image_path][1][1]

        event_probabilities[time_index, cdp_index] = prob_event/100










# now display the image
plt.imsave('images/output/experimento0--{}.png'.format(timestamp), event_probabilities, cmap='gray')

plt.imshow(event_probabilities) # default color map interpolates between 0.0 and 1.0
plt.show()





