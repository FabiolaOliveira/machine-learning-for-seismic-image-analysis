#!/usr/bin/env python3

# script requires a unix-like environment


"""
experiment0 involves the following steps:

    1. take an image (CMP stacked) of seismic amplitudes
    2. divide it up into 2k+1 by 2k+1 windows
    3. some of the windows for we which we have a class label
        ('event' or 'no event' depending on the classification of the central pixel)
        will be part of the training/validation set.
    4. all the other windows of the entire image will be part of the test set.
    5. we'll send POST requests to a running instance of an NVIDIA DIGITS server in order to:
        a) create a dataset
        b) create a model
        c) classify the test set, stashing the results
    6. since we don't necessarily have class labels for the test set,
        we will simply generate an image with a color map that reflects
        the probability of event/noevent at each pixel.
    7. we display the image
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread as scipy_imread, imsave as scipy_imsave
# NOTE: SCIPY'S IMREAD IS SAVING IMAGES WITH AN UNEVEN COLOR MAP!!! (I.E. DIFFERENT FOR EACH IMAGE). CORRECT THIS

import sys
import json
import re
import argparse
import os

import pymei as pm
from proj_utilities import run_command, print_command_info, load_seismic_image, load_seismic_picks_json_format, NVDigits_API


class Experiment0():
    """used to run instances of experiment0"""


    def __init__(self, args, nvdigits_host='localhost', nvdigits_port='80'):
        """
        prepare experiment0 related processing. given parsed command line arguments object
        and location where NVIDIA digits is running

        """

        self.CODE_FOLDER = os.path.realpath(sys.path[0]) # obtain the full path of the script folder
        self.LOG_FOLDER = self.CODE_FOLDER + '/../logs/'
        self.INPUT_FOLDER = self.CODE_FOLDER + '/../inputs/'
        self.OUTPUT_FOLDER = self.CODE_FOLDER + '/../outputs/'
        self.GENERAL_LOG = self.CODE_FOLDER + '/../general_log'


        # if window size isn't odd, there is no central pixel
        if args.window_size < 0 or args.window_size % 2 != 1:
            raise ArgumentError("must specify a valid odd window size argument")
        self.half_window_size = args.window_size // 2
        self.window_size = args.window_size

        self.username = run_command('whoami')['stdout'].replace('\n','') 

        # name given to this run of the experiment
        self.name = args.name or ('experiment0--' + self.username)

        # identify this run of the experiment
        self.timestamp = run_command('date "+%Y-%m-%d--%H-%M-%S"')['stdout'].replace('\n','')

        # print diagnostic information to log file
        self.log = args.log or (self.LOG_FOLDER + self.name + '--' + self.timestamp)
        self.log_file = open(self.log, 'w')
        self.log_file.write(self.name + '\n\n' + self.timestamp + '\n\nlogging this instance of the experiment:\n\n')

        self.general_log_file = open(self.GENERAL_LOG, 'a')
        self.general_log_file.write('-'*100 + '\n' + self.name + '\n\n' + self.timestamp + '\n\nlogging this instance of the experiment:\n\n')

        # flag for printing diagnostic information to stdout
        self.verbose = args.verbose


        if args.image.endswith('sgy') or args.image.endswith('su'):
            self.traces, self.img = load_seismic_image(args.image)
        elif args.image.endswith('png'):
            self.img = scipy_imread(
                fname=args.image,
                flatten=True) # obtains a gray scale image
        else:
            raise Exception('unknown format: only sgy, su and png accepted')

        # maximum sample value, to be used in color maps
        self.vmax = np.max(np.abs(self.img))

        # minimum value to be used in color maps
        self.vmin = -self.vmax # preferred since it maps zero to zero
        #self.vmin = np.min(img)

        # set of input training points displayed in (CDP, time) columns
        self.no_event_points = load_seismic_picks_json_format(args.no_event_picks)
        self.event_points = load_seismic_picks_json_format(args.event_picks)


        # our interface to NVIDIA's software for demonstrating deep learning computations on a GPU (by way of BVLC Caffe)
        self.nvdigits = NVDigits_API(nvdigits_host, nvdigits_port)

        # necessary for logging in to the server for certain operations
        self.cookie_file = args.cookie_file

        # path where training images will be placed:
        self.training_folder = os.path.realpath(args.training_folder or (self.INPUT_FOLDER + 'train--' + self.name + '--' + self.timestamp))

        # path where the test images will be placed:
        self.test_folder = os.path.realpath(args.test_folder or (self.INPUT_FOLDER + 'test--' + self.name + '--' + self.timestamp))

    # context manager protocol is used since we manage an open log file
    def __enter__(self):
        return self
    def __exit__(self, ex_type, ex_value, traceback):
        if ex_type is not None: # if we're closing with an exception
            self.general_log_file.write('closing this instance with an exception:\n' + str(ex_type) + '\n' + str(ex_value) + '\n' + str(traceback) + '\n')
            self.log_file.write('closing this instance with an exception:\n' + str(ex_type) + '\n' + str(ex_value) + '\n' + str(traceback) + '\n')
        else:
            self.general_log_file.write('successfully executed this instance of experiment0' + '\n' + '-'*100 + '\n'*10)
            self.log_file.write('successfully executed this instance of experiment0')
        self.log_file.close()
        self.general_log_file.close()
        return False # keep propagating exception, if applicable


    def obtain_cookie(self):
        """ensure that we have a cookie to login to the nvidia digits server for certain operations"""
        
        self.log_message('obtaining cookie from nvidia digits server')

        if run_command('ls ' + self.cookie_file)['returncode'] != 0: 
            # login cookie not found, so bake it:
            bake_cookie_command = self.nvdigits.bake_cookie(self.cookie_file, self.username)
            
            # log our activity
            print_command_info(bake_cookie_command, file=self.log_file)
            if self.verbose:
                print_command_info(bake_cookie_command)

        self.nvdigits.set_cookie_file(self.cookie_file)

    def generate_training_images(self):
        """
        generates folder with training images for the experiment
        folder where the training images will be placed is split into subfolders whose names are the class labels.
        """

        self.log_message('obtaining images for training')

        run_command('mkdir -p ' + self.training_folder)
        run_command('mkdir -p ' + self.training_folder + '/noevent')
        run_command('mkdir -p ' + self.training_folder + '/event')

        # Data with 'non events' followed by labeled seismic events:
        for example_points, class_label in [(self.no_event_points, 'noevent'), (self.event_points, 'event')]:
            
            # each training point (pick) is of the form [time sample index, trace index]
            for i, pt in enumerate(example_points):
                time_coord, trace_index = pt[0], pt[1]

                # warning issued if any point is too close to the border. validity of the model of 'central pixel' called into question
                if (time_coord-self.half_window_size<0 or 
                    time_coord+self.half_window_size>=self.img.shape[0] or 
                    trace_index-self.half_window_size<0 or 
                    trace_index+self.half_window_size>=self.img.shape[1]):

                    self.log_file.write('WARNING: the window for point {} of type {} overflows image bounds\n'.format(i, class_label))
                    if self.verbose:
                        print('WARNING: the window for point {} of type {} overflows image bounds'.format(i, class_label))

                tmin = max(0, time_coord-self.half_window_size)
                tmax = min(self.img.shape[0], time_coord+self.half_window_size+1)
                cdpmin = max(0, trace_index-self.half_window_size)
                cdpmax = min(self.img.shape[1], trace_index+self.half_window_size+1)

                window = self.img[tmin:tmax, cdpmin:cdpmax]

                # plt seems to insist on 4 color channels! scipy.misc on the other hand seems to just normalize and save the intensities
                scipy_imsave('{}/{}/image{}-time{}-trace{}.png'.format(self.training_folder, class_label, i, time_coord, trace_index), window)
                #plt.imsave('{}/{}/image{}-time{}-trace{}.png'.format(self.training_folder, class_label, i, time_coord, trace_index), window, cmap='gray', vmin=self.vmin, vmax=self.vmax)

    def create_dataset(self):
        self.log_message('obtaining dataset from nvidia digits server')

        create_dataset_command = self.nvdigits.create_dataset(
            folder_train=self.training_folder, # contains subfolders whose names are the class labels and whose contents are the training images
            encoding='png',
            resize_channels='1', # grayscale image: 1 color channel
            resize_height=str(self.window_size),
            resize_width=str(self.window_size),
            dataset_name=self.name + '--' + self.timestamp,
        )
        print_command_info(create_dataset_command, file=self.log_file)
        if self.verbose:
            print_command_info(create_dataset_command) #print(json.dumps(create_dataset, indent=4))
        
        # we'll use this id to refer to the dataset in future operations
        self.dataset_id = json.loads(create_dataset_command['stdout'])['id']
        
        # wait until the dataset is ready
        query_command = self.nvdigits.wait_until_dataset_is_ready(self.dataset_id)
        print_command_info(query_command, file=self.log_file)
        if self.verbose:
            print_command_info(query_command)

        if self.verbose:
            # query more information about the dataset
            query_command = self.nvdigits.query_dataset(self.dataset_id)
            print_command_info(query_command)
            print_command_info(query_command, file=self.log_file)
            
            # also return some metadata about datasets and classification models stored in the DB
            query_command = self.nvdigits.models_and_datasets()
            print_command_info(query_command)
            print_command_info(query_command, file=self.log_file)

    def log_message(self, message):
        if self.verbose:
            print('\n' + message)
        self.log_file.write('\n' + message + '\n')

    def create_model(self):
        self.log_message('obtaining model from nvidia digits server')

        create_model_command = self.nvdigits.create_model(
            method='standard',
            standard_networks='lenet',
            framework='caffe',
            train_epochs='30',
            solver_type='SGD',
            learning_rate='0.01',
            use_mean='image',
            # random_seed='42' # set a seed to reproduce training results for multiple runs
            dataset=self.dataset_id,
            model_name=self.name + '--' + self.timestamp
        )
        print_command_info(create_model_command, file=self.log_file)
        if self.verbose:
            print_command_info(create_model_command)

        # we'll use this id to refer to the model in future operations
        self.model_id = json.loads(create_model_command['stdout'])['id']

        query_command = self.nvdigits.wait_until_model_is_ready(self.model_id, wait_hook_function=self.log_message if self.verbose else None)
        print_command_info(query_command, file=self.log_file)
        if self.verbose:
            print_command_info(query_command)

        # return some metadata about datasets and classification models stored in the DB
        if args.verbose:
            query_command = self.nvdigits.models_and_datasets()
            print_command_info(query_command)
            print_command_info(query_command, file=self.log_file)

    def generate_test_images(self):
        self.log_message('obtaining images for testing')

        run_command('mkdir -p ' + self.test_folder)

        for trace_index in range(self.half_window_size, self.img.shape[1]-self.half_window_size):
            # one separate test image folder for each seismic trace/time series
            test_image_folder = '{}/trace{}'.format(self.test_folder, trace_index)
            run_command('mkdir -p ' + test_image_folder)
            image_paths_file = '{}/image_paths.txt'.format(test_image_folder) # file with the path of test images to be classified
            
            with open(image_paths_file, 'w') as listofimagepaths:
                for time_index in range(self.half_window_size, self.img.shape[0]-self.half_window_size):
                    
                    window = self.img[time_index - self.half_window_size  :  time_index + self.half_window_size+1, trace_index - self.half_window_size  :  trace_index + self.half_window_size+1]
                    
                    image_filename = '{}/time{}-trace{}.png'.format(test_image_folder, time_index, trace_index)
                    
                    # plt seems to insist on 4 color channels! scipy.misc on the other hand seems to just normalize and save the intensities
                    scipy_imsave(image_filename, window)
                    # plt.imsave(image_filename, window, cmap='gray', vmin=self.vmin, vmax=self.vmax)
                    
                    listofimagepaths.write('trace{}/time{}-trace{}.png'.format(trace_index, time_index, trace_index) + '\n')

            if self.verbose:
                print('finished with test images folder {}'.format(test_image_folder))

    def classify_images(self):
        self.log_message('obtaining classification from nvidia digits server')

        for trace_index in range(self.half_window_size, self.img.shape[1]-self.half_window_size):
            # one separate test image folder for each cdp (seismic trace/time series)
            test_image_folder = '{}/trace{}'.format(self.test_folder, trace_index)
            image_paths_file = '{}/image_paths.txt'.format(test_image_folder) # file with the path of test images to be classified
            predictions_file = '{}/predictions'.format(test_image_folder)
            success = False
            while not success:
                c = self.nvdigits.classify_many(
                    model_id=self.model_id,
                    image_folder=self.test_image_folder,
                    image_list=image_paths_file # each line has the path for a test image
                    #predictions_file=predictions_file
                )
                try:
                    cl = json.loads(c['stdout'])['classifications']
                    if cl:
                        success = True
                    with open(predictions_file, 'w') as predfile:
                        predfile.write(c['stdout'])
                except Exception as e:
                    print_command_info(c, file=self.log_file)
                    print(e, file=self.log_file)
                    if self.verbose:
                        print_command_info(c)
                        print(e)
                    # we failed; try to classify the images again

            print('commandline:', c['commandline'], '\nreturncode:', c['returncode'], '\nstderr:', c['stderr'], sep='\n', file=self.log_file)
            if self.verbose:
                print('commandline:', c['commandline'], '\nreturncode:', c['returncode'], '\nstderr:', c['stderr'], sep='\n')

    def generate_probability_image(self):
        """now prepare the image for display with a simple color map from 0 to 100 % chance of seismic event at each pixel, as predicted above"""
        
        self.log_message('generating class probability image from the classified test set')

        self.event_probabilities = np.zeros(self.img.shape)
        time_index_pattern = re.compile(r'time([0-9]+)-trace[0-9]+\.png$') # extract the time index from the image path

        for trace_index in range(self.half_window_size, self.img.shape[1]-self.half_window_size):
            # one separate test image folder for each seismic trace/time series
            test_image_folder = '{}/trace{}'.format(self.test_folder, trace_index)
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

                self.event_probabilities[time_index, trace_index] = prob_event/100.0

        return self.event_probabilities

    def save_probability_image(self, cmap='gray'):
        self.log_message('saving obtained class probability image')

        plt.imsave(self.OUTPUT_FOLDER + '{}--{}.png'.format(self.name, self.timestamp), self.event_probabilities, cmap=cmap)






if __name__ == '__main__':
    # process command line options

    # description is the docstring above
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=globals()['__doc__'])

    parser.add_argument('-w', '--window-size', type=int, default=33, help="define the (odd) size, 2K+1 by 2K+1, of each cutout of the main image")
    parser.add_argument('-n', '--name', default='', help='give this dataset/model/classification/image generation experiment a name. default is experiment0-<your username>')
    parser.add_argument('-l', '--log', default='', help="log all console output to this file. default is <name>--<timestamp>")
    parser.add_argument('-v', '--verbose', action='store_true', help='print diagnostic info')
    parser.add_argument('-c', '--cookie-file', default='digits.cookie', help='cookie file for NVIDIA DIGITS server (used for login)')

    parser.add_argument('-gt', '--generate-training', action='store_true', help='the training images will be generated from the input image')
    parser.add_argument('-tf', '--training-folder', default='', help='folder where the training images are located (split into subfolders whose names are the class labels)')
    
    parser.add_argument('-cd', '--create-dataset', action='store_true', help='create an NVIDIA Digits dataset with training images')
    parser.add_argument('-d', '--dataset', default='', help="specify DIGITS dataset name to use")
    
    parser.add_argument('-cm', '--create-model', action='store_true', help='create an NVIDIA Digits model with training images')
    parser.add_argument('-m', '--model', default='', help="specify DIGITS model name to use")
    
    parser.add_argument('-gtst', '--generate-test', action='store_true', help='generate test images from the input image')
    parser.add_argument('-tstf', '--test-folder', default='', help='folder where the test images are')
    
    parser.add_argument('-cl', '--classify', action='store_true', help='classify the images in the test folder, generating class probabilities for each')
    
    parser.add_argument('-o', '--output-image', default='', help='path of probability map image to be saved. default is <name>--<timestamp>')

    parser.add_argument('-ne', '--no-event-picks', default='', help='file with time sample and trace indices of selected "no event" points in the image')
    parser.add_argument('-e', '--event-picks', default='', help='file with time sample and trace indices of selected "event" points in the image')
    parser.add_argument('-i', '--image', help='SEGY, SU, or PNG image on which to perform experiment0')
    
    args = parser.parse_args()


    # CARRY OUT EXPERIMENT 0

    with Experiment0(args) as exp:

        if args.verbose: 
            print('initializing experiment0')
        
        # requires NVIDIA DIGITS
        if args.verbose: 
            print('obtaining cookie from nvidia digits server')
        exp.obtain_cookie()

        # requires source image file and event + no event picks
        if args.generate_training:
            if args.image is '' or args.event_picks is '' or args.no_event_picks is '':
                exp.log_message('Unable to proceed:\n' 
                    'to generate training images, a source image,'
                    ' event picks and no-event picks are required'
                )
                if args.verbose:
                    print('Unable to proceed:\n' 
                        'to generate training images, a source image,'
                        ' event picks and no-event picks are required'
                    )
                sys.exit(-1)
            if args.verbose: 
                print('generating images for training')
            exp.generate_training_images()

        if args.create_dataset:
            # requires NVIDIA DIGITS and training images divided into subfolders whose names are the class labels
            if args.generate_training is False and args.training_folder is '':
                exp.log_message('Unable to proceed:\n' 
                    'to generate a dataset, a folder with training images is required'
                )
                if args.verbose:
                    print('Unable to proceed:\n' 
                        'to generate a dataset, a folder with training images is required'
                    )
                sys.exit(-1)        
            if args.verbose: 
                print('creating dataset at nvidia digits server')
            exp.create_dataset()
        elif args.dataset is not '':
            # requires NVIDIA DIGITS
            if args.verbose: 
                print('obtaining dataset from nvidia digits server')
            exp.dataset_id = exp.nvdigits.get_dataset_id_by_name(args.dataset)


        if args.create_model:
            # requires NVIDIA DIGITS and a dataset
            if args.create_dataset is False and args.dataset is '':
                exp.log_message('Unable to proceed:\n' 
                    'to generate a model, an NVIDIA DIGITS dataset is required'
                )
                if args.verbose:
                    print('Unable to proceed:\n' 
                        'to generate a model, an NVIDIA DIGITS dataset is required'
                    )
                sys.exit(-1)
            if args.verbose: 
                print('creating model from nvidia digits server')
            exp.create_model()
        elif arg.model != '':
            # requires NVIDIA DIGITS
            if args.verbose: 
                print('obtaining model from nvidia digits server')
            exp.model_id = exp.nvdigits.get_model_id_by_name(args.model)


        if args.generate_test is not '':
            # requires source image file
            if args.image is '':
                exp.log_message('Unable to proceed:\n' 
                    'to generate test images, a source image is required'
                )
                if args.verbose:
                    print('Unable to proceed:\n' 
                        'to generate test images, a source image is required'
                    )
                sys.exit(-1)
            if args.verbose: 
                print('generating images for testing')
            exp.generate_test_images()


        if args.classify:
            # requires NVIDIA DIGITS and a test folder
            if args.generate_test is False and args.test_folder is '':
                exp.log_message('Unable to proceed:\n' 
                    'to classify test images, a folder with those images is required'
                )
                if args.verbose:
                    print('Unable to proceed:\n' 
                        'to classify test images, a folder with those images is required'
                    )
                sys.exit(-1)
            if args.verbose: 
                print('obtaining classification from nvidia digits server')
            exp.classify_images()

        # requires a test folder with a predictions file in every subfolder corresponding to a trace
        if args.generate_test is False and args.test_folder is '':
            exp.log_message('Unable to proceed:\n' 
                'to classify test images, a folder with those images is required'
            )
            if args.verbose:
                print('Unable to proceed:\n' 
                    'to classify test images, a folder with those images is required'
                )
            sys.exit(-1)
        if args.verbose: 
            print('generating class probability image from the classified test set')
        img = exp.generate_probability_image()

        if args.verbose: 
            print('saving obtained class probability image')
        exp.save_probability_image(cmap='gray')

    # now display the image
    plt.imshow(img, cmap='gray')
    plt.show()
