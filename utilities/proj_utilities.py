import subprocess
import json
import re
import sys

import pymei as pm
import numpy as np
import matplotlib.pyplot as plt

"""

"""

"""shell utils"""

def run_command(commandline, sendtostdin=None):
    """
    Execute a shell command line, wait for its completion, and return information about its execution.

    Parameters
    ----------
    commandline : str
        This command string is executed by /bin/sh.
    sendtostdin : str, optional
        This string is piped to the standard input of the command.

    Returns
    -------
    command_info : dict
        Dictionary with the 'commandline', 'returncode', 'stdout' and 'stderr' keys

    """
    
    if type(commandline) is not str or (sendtostdin is not None and type(sendtostdin) is not str):
        raise TypeError('must pass in a string as command line to be executed in the shell, and, optionally, a string to send to the process\'s stdin')
    
    proc = subprocess.Popen(
        commandline, 
        universal_newlines=True, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    out, err = proc.communicate(sendtostdin) # will wait until the process is finished
    
    return {
        'commandline': commandline,
        'returncode': proc.returncode,
        'stdout': out,
        'stderr': err
    }

def print_command_info(command_object, spaces=4, file=sys.stdout):
    """
    Pretty-print the information in a dictionary returned by run_command

    Parameters
    ----------
    command_object : dict
        Dictionary with the 'commandline', 'returncode', 'stdout' and 'stderr' keys containing information about an executed command.
    spaces : int, optional
        Number of spaces of indentation for each line of the printout.
    file : file
        Print to this file

    """
    
    if type(command_object) is not dict:
        raise TypeError('print_command_info expects a dictionary with the information about a command, but you provided a' + type(command_object))
    
    print(
        '{',
        '\n',
        spaces*' ', '"commandline":', '\n',
        # we replace literal newlines in the files with actual newlines to cover the case of JSON output from commands
        spaces*' ', command_object['commandline'].replace('\\n', '\n').replace('\n', '\n' + spaces*' '), '\n',
        '\n',
        '"returncode":', '\n',
        command_object['returncode'], '\n',
        spaces*' ', '"stdout":', '\n',
        spaces*' ', command_object['stdout'].replace('\\n', '\n').replace('\n', '\n' + spaces*' '), '\n',
        '\n',
        spaces*' ', '"stderr":', '\n',
        spaces*' ', command_object['stderr'].replace('\\n', '\n').replace('\n', '\n' + spaces*' '), '\n',
        '\n',
        '}',

        sep='', file=file
    )


def strip_path_from_filenames(list_of_files):
    """Strip the path from each filename in a list of filenames.

    Parameters
    ----------
    list_of_files : sequence
        Iterable sequence of filenames

    Returns
    -------
    stripped_filenames : list
        List of filenames with the leading directory components stripped.

    """

    select_basename = re.compile(r'/?([^/]*)$') # capture basename
    return [re.search(select_basename, f)[1] for f in list_of_files]


"""some image processing utilities"""

def convert_gray_image_to_uint8(img):
    """converts a numpy array of arbitrary values to values in [0, 255]

    Parameters
    ----------
    img : ndarray
        An array of gray image intensity values

    Returns
    -------
    img_uint8
        An array of uint8 obtained from img via a linear transformation
        followed by rounding
    """

    minimum = np.min(img)
    maximum = np.max(img)

    return np.array(np.round(255 *
                            ((img - minimum) / (maximum - minimum))),
                    dtype=np.uint8)


"""seismic image utils"""


def load_seismic_image(filename):
    """Read traces from an SU or SEGY file into a list of trace objects and a numpy array.

    Parameters
    ----------
    filename : str
        name of SU/SEGY file to load

    Returns
    -------
    (traces, data_img) : tuple
        traces : list
            list of pymei Trace objects corresponding to each seismic trace in the file (header and payload)
        data_img : ndarray
            numpy ndarray of shape (ns, len(traces)) containing the data in the file
            indexed by (time, trace_index). trace_index is given by the order obtained from the file.
            The data is disposed in column major (fortran contiguous) order.

    Notes
    -----
    pymei.load returns a SEGY/SU object containing data from the SEGY/SU file.
    SEGY, for example, has a textual header (3200 byte EBCDIC), a binary header (400 byte), 
    and a header+payload for each seismic trace (header is 240 bytes, payload is 4*ns bytes) 
    (there are ns samples in a trace, each one being a float (possibly an 'IBM float'))

    See Also
    --------
    pymei.load
    
    """

    traces = [trace for trace in pm.load(filename)]
    data_img = np.array([trace.data for trace in traces]).T

    return traces, data_img


def load_seismic_picks_CDP_format(filename, traces):
    """Read seismic image picks from a text file in a legacy format (CDP, time (ms))

    Parameters
    ----------
    filename : str
        Text file with the picks. format is given by:

        (headers for illustration only. file is only lines with 2 space-separated floats):

        CDP number | Time (ms)
        ----------------------
        ...        | ...
        ...        | ...

    traces : list
        list of pymei Trace objects.

    Returns
    -------
    pick_img_coords : list
        List of 2 item lists of the form (time index, trace index). 
        Each of them is a pick in the data image formed by disposing the traces as columns.

       

    """
    
    pick_file_coords = {int(cdp): time for cdp, time in np.loadtxt(filename)}
    pick_img_coords = []

    # have to run through since we don't really know if CDP indices are in order, what their starting index is, etc...
    for trace_index, trace in enumerate(traces): 
        # convert the CDP number into a trace index
        # convert ths time (in ms) into a (time) sample index in the trace
        if trace.cdp in pick_file_coords: # found one
            # convert the time to seconds, then divide by sample interval size to find the time sample index
            pick_img_coords.append(
                [
                    int(pick_file_coords[trace.cdp] / 1000 / trace.dt),
                    trace_index
                ]
            )

    # sanity checks
    if len(pick_file_coords) > len(pick_img_coords):
        raise Exception('could not find some picked CDP number in the list of traces')
    elif len(pick_file_coords) < len(pick_img_coords):
        raise Exception('repeated CDP found in list of traces')

    return pick_img_coords

def load_seismic_picks_json_format(fname):
    """read seismic picks from a file in a JSON format.

    The JSON structure is an object with a key named 'picks',
    whose value is an array of length 2 arrays of the form
    [time sample index, trace index] representing a selected point
    in a seismic image.

    Parameters
    ----------
    fname : str
        Text file containing a JSON structure with the picks.

    Returns
    -------
    pick_coordinates : list
        list of lists of the form [time sample index, trace index]
        found in the file

    """

    with open(fname, 'r') as f:
        pick_coordinates = json.load(f)['picks']

    return pick_coordinates

def print_trace_headers(fname, file=sys.stdin):
    """print data from each trace header in a SEGY/SU file.

    Parameters
    ----------
    fname : str
        path to SU/SEGY file
    file : file
        output stream where information will be placed

    """

    traces, img = load_seismic_image(fname)

    # table header with the important trace attributes
    print('%10s %10s %10s %10s  %12s %12s %12s %12s %12s %12s %12s %12s'
      % ('index', 'cdp', 'ns', 'dt', 'gx', 'gy', 'sx', 'sy', 'mx', 'my', 'hx', 'hy'), file=file)

    # table row for the header information each trace
    for i, trace in enumerate(traces):
        print('%10i %10i %10i %10.6f  %12f %12f %12f %12f %12f %12f %12f %12f'
              % (i, trace.cdp, trace.ns, trace.dt, 
                 trace.gx, trace.gy, trace.sx, trace.sy,
                 trace.mx, trace.my, trace.hx, trace.hy), file=file)
    print('\n\n\n', file=file)

def print_trace_info(fname, file=sys.stdin):
    """print data from each trace in a SEGY/SU file.

    Parameters
    ----------
    fname : str
        path to SU/SEGY file
    file : file
        output stream where information will be placed

    """

    traces, img = load_seismic_image(fname)

    # print the header
    for i in range(len(traces)):
        print('%10s' % i, end='', file=file)
    print('\n')
    for trace in traces:
        print('%10s' % trace.cdp, end='', file=file)
    print('\n')

    # print the data from each each trace (traces are columns)
    for time in range(img.shape[0]):
        for trace in range(img.shape[1]):
            print('%10f ' % img[time, trace], end='', file=file)
        print('\n')



"""NVIDIA DIGITS"""

class NVDigits_API():
    """
    Interface, via a RESTful API, to NVIDIA Digits:
    NVIDIA's software for demonstrating deep learning computations on a GPU (by way of BVLC Caffe).
    
    Place requests regarding models and datasets; receive a JSON encoded response.
    Methods return a command_info dictionary 
    ('commandline', 'returncode', 'stdout', and 'stderr' of placed commands).

    State: host and port where NVIDIA Digits should be running
           cookie file for authentication with the server (necessary for certain operations)
    
    See Also
    --------
    https://github.com/NVIDIA/DIGITS/blob/be2bc827cab5a7bc3f9c2e5baa9d8cd48fd414eb/docs/API.md
    (for some examples of the digits API ('docs' folder, master branch of the github repository)

    TO DO: error checking, iterating over datasets/models, displaying model statistics, etc.

    """

    def __init__(self, host, port):
        """Precondition: Nvidia Digits should be running on host:port"""
        
        self.host = host
        self.port = port

        # add at the beginning of every call
        self.call_curl = 'curl ' + self.host + ':' + self.port

    def bake_cookie(self, cookie_file, username):
        """Create a cookie file for user at the given pathname."""

        bake_cookie_command = (
            self.call_curl + 
            '/login' +
            ' -c ' + cookie_file + 
            ' -XPOST' +
            ' -F username=' + username
        )
        bkc = run_command(bake_cookie_command)
        return bkc

    def set_cookie_file(self, cookie_file):
        self.cookie_file = cookie_file

    def create_dataset(self, folder_train, encoding, resize_channels, resize_height, resize_width, dataset_name):
        create_dataset_command = (
            self.call_curl + 
            '/datasets/images/classification.json' +
            ' -b ' + self.cookie_file + # must login to create a dataset
            ' -XPOST' + # use POST method
            ' -F folder_train=' + folder_train +
            ' -F encoding=' + encoding +
            ' -F resize_channels=' + resize_channels +
            ' -F resize_height=' + resize_height + 
            ' -F resize_width=' + resize_width + 
            ' -F method=folder' +
            ' -F dataset_name=' + dataset_name
        )
        create_dataset = run_command(create_dataset_command)
        return create_dataset

    def create_model(self, method, standard_networks, framework, train_epochs, solver_type, learning_rate, use_mean, dataset, model_name):
        # create a new classification model
        create_model_command = (
            self.call_curl +
            '/models/images/classification.json' +
            ' -b ' + self.cookie_file +
            ' -XPOST' +
            ' -F method=' + method + # use one of the standard networks: LeNet, AlexNet, GoogLeNet
            ' -F standard_networks=' + standard_networks + # the more basic, time-honored convolutional neural network
            ' -F framework=' + framework +
            ' -F train_epochs=' + train_epochs + # number of passes through the training data
            ' -F solver_type=' + solver_type + # stochastic gradient descent
            ' -F learning_rate=' + learning_rate + # step size during gradient descent
            ' -F use_mean=' + use_mean + # subtracts mean image from the data
            # ' - F random_seed=42' # set a seed to reproduce training results for multiple runs
            ' -F dataset=' + dataset +
            ' -F model_name=' + model_name
        )
        create_model = run_command(create_model_command)
        return create_model

    def classify_many(self, model_id, image_folder, image_list): #, predictions_file):
        classify_many_command =(self.call_curl + '/models/images/classification/classify_many.json' +
                                ' -XPOST' +
                                ' -F job_id=' + model_id +
                                ' -F image_folder=' + image_folder + # prepend this path to the image paths in the list
                                ' -F image_list=@' + image_list) # each line has the path for a test image
        c = run_command(classify_many_command)
        return c

    def wait_until_dataset_is_ready(self, dataset_id, wait_hook_function=None):
        """
        Loop until dataset is ready. 
        Return information from the command that obtains final dataset status.
        wait_hook_function, if provided, is called with a wait message
        (it could throw an exception to abort the wait, for example)
        """
        # TODO: error checking
        waiting = True
        while waiting:
            run_command('sleep 1')
            q = self.query_dataset_status(dataset_id)
            if json.loads(q['stdout'])['status'] == 'Done':
                waiting = False # dataset has been successfully created
            elif wait_hook_function is not None:
                wait_hook_function('waiting for dataset creation to finish. please remain calm')
        return q

    def wait_until_model_is_ready(self, model_id, wait_hook_function=None):
        """
        Loop until model is ready. 
        Return information from the command that obtains final model status.
        wait_hook_function, if provided, is called with a wait message
        (it could throw an exception to abort the wait, for example)
        """
        # TODO: error checking
        waiting = True
        while waiting:
            run_command('sleep 10')
            q = self.query_model_status(model_id)
            if json.loads(q['stdout'])['status'] == 'Done':
                waiting = False # dataset has been successfully created
            elif wait_hook_function is not None:
                wait_hook_function('waiting for model to finish running. please remain calm')
        return q

    def get_dataset_id_by_name(self, name):
        """Take the name of a dataset and return its id"""

        dataset_id = None
        q = run_command(self.call_curl + '/index.json')
        for m in json.loads(q['stdout'])['datasets']:
            if m['name'] == name:
                dataset_id = m['id']
                break
        return dataset_id

    def get_model_id_by_name(self, name):
        """Take the name of a model and return its id"""

        model_id = None
        q = run_command(self.call_curl + '/index.json')
        for m in json.loads(q['stdout'])['models']:
            if m['name'] == name:
                model_id = m['id']
                break
        return model_id

    def query_dataset_status(self, dataset_id):
        """Return JSON encoded information about a dataset's status"""

        q = run_command(self.call_curl + '/datasets/' + dataset_id + '/status')
        return q

    def query_model_status(self, model_id):
        """return JSON encoded information about a dataset's status"""

        q = run_command(self.call_curl + '/models/' + model_id + '.json')
        return q

    def query_dataset(self, dataset_id):
        """Return JSON encoded information about a dataset."""

        q = run_command(self.call_curl + '/datasets/' + dataset_id + '.json') 
        return q

    def models_and_datasets(self):
        """Return JSON encoded information about models and datasets in the database."""

        q = run_command(self.call_curl + '/index.json')
        return q



"""unit tests"""

if __name__ == '__main__':
    c = run_command('grep -n -r -P "\b\w{15,}\b"') # find file:lines in the directory tree with words having length at least 15
    print_command_info(c)



