
#############################################################################
# Copyright 2011 Jason Baldridge
# 
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#############################################################################

import math,sys,tempfile,os
from optparse import OptionParser
from itertools import islice

# Read in data from a file in the format:
#   Attr1=Val1,Attr2=Val2,...,AttrN=ValN,Label
#
# Returns a list of pairs, where each pair has the attribute-value
# list as its first element and the associated label as its second.
def read_data (inputFileName):
    alldata = []
    for line in inputFileName.readlines():
        info = line.strip().split(',')
        alldata.append(((info[:-1]),info[-1]))
    return alldata

# Print an error message if verbose output is desired
def errmsg (msg, verbose):
    if verbose:
        print(msg,file=sys.stderr)

# Make sure all mandatory options appeared.
def check_mandatory_options (parser, options, mandatories):
    for m in mandatories:
        if not options.__dict__[m]:
            print("\nMandatory option '" + m + "' is missing.\n")
            parser.print_help()
            exit(-1)

# Make sure that meta characters "," and "=" are protected so that
# things like "50,000" aren't thought to be parts of two different
# features. 
def protect_meta_characters (text):
    return text.replace(",", "COMMA").replace("=","EQUALS")

def makefeat(attribute, value):
    return attribute+"="+protect_meta_characters(value)

# Create option parser for categorization programs.
def get_categorizer_option_parser ():
    parser = OptionParser()
    parser.add_option("-t", "--train", dest="train",
                      help="learn from training events in FILE", metavar="FILE")
    parser.add_option("-p", "--predict", dest="predict",
                      help="predict labels for events in FILE", metavar="FILE")
    parser.add_option("-o", "--out", dest="out",
                      help="output predictions to FILE", metavar="FILE")
    parser.add_option("-l", "--lambda", dest="lambda_value", type="float", default=0,
                      help="lambda value used in smoothing", metavar="FLOAT")
    parser.add_option("-v", "--verbose",
                      action="store_true", dest="verbose", default=False,
                      help="print status messages to stderr")
    return parser

def get_twitter_option_parser ():
    parser = OptionParser()
    parser.add_option("-t", "--train", dest="train",
                      help="learn from training events in FILE", metavar="FILE")
    parser.add_option("-a", "--auxtrain", dest="auxtrain",
                      help="use auxiliary training events in FILE in addition to those specified with --train", metavar="FILE")
    parser.add_option("-e", "--eval", dest="eval",
                      help="predict labels for events in FILE", metavar="FILE")
    parser.add_option("-o", "--out", dest="out",
                      help="output predictions to FILE", metavar="FILE")
    parser.add_option("-m", "--modeltype", dest="model_type", default="mx",
                      help="type of model to use: maj (majority class baseline), lex (lexicon ratio baseline), nb (naive Bayes), pt (perceptron), mx (maximum entropy, default)", metavar="MODELTYPE")
    parser.add_option("-r", "--sthreshold", dest="subjectivity_threshold", type="float", default=0.5,
                      help="during subjectivity classification, the confidence of the classifier must be greater than this value in order for the item to be labeled as subjective (default = 0.5)", metavar="FLOAT")
    parser.add_option("-p", "--pthreshold", dest="polarity_threshold", type="float", default=0.3,
                      help="during polarity classification, the confidence of the classifier must be greater than this value in order for the item to be labeled at all (default = 0.3)", metavar="FLOAT")
    parser.add_option("-s", "--smoothing", dest="smoothing_value", type="float", default=1.0,
                      help="value used in smoothing, either lambda for naive Bayes or sigma for maxent (default = 1.0)", metavar="FLOAT")
    parser.add_option("-w", "--twostage",
                      action="store_true", dest="twostage", default=False,
                      help="do subjectivity classification and then polarity classification")
    parser.add_option("-x", "--extendedfeatures",
                      action="store_true", dest="extended_features", default=False,
                      help="use extended features")
    parser.add_option("-v", "--verbose",
                      action="store_true", dest="verbose", default=False,
                      help="print status messages to stderr")
    parser.add_option("-d", "--detail",
                      action="store_true", dest="detailed_output", default=False,
                      help="print detailed output regarding correct and incorrect classifications")
    return parser

def get_feature_extractor_option_parser ():
    parser = OptionParser()
    parser.add_option("-i", "--input", dest="input",
                      help="read raw data from FILE", metavar="FILE")
    parser.add_option("-o", "--out", dest="out",
                      help="output features to FILE", metavar="FILE")
    parser.add_option("-v", "--verbose",
                      action="store_true", dest="verbose", default=False,
                      help="print status messages to stderr")
    parser.add_option("-e", "--extended_features",
                      action="store_true", dest="extended_features", default=False,
                      help="output extended features")
    return parser

def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result    
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def train_and_classify (training_events, eval_events, model_type, smoothing_parameter, verbose=False):

    training_events_file = tempfile.NamedTemporaryFile()
    for event in training_events:
        training_events_file.write((','.join(event) + "\n").encode())
    training_events_file.flush()

    eval_events_file = tempfile.NamedTemporaryFile()
    for event in eval_events:
        eval_events_file.write((','.join(event) + "\n").encode())
    eval_events_file.flush()

    if model_type == "nb":
        return run_naivebayes(training_events_file.name, eval_events_file.name,
                              smoothing_parameter, verbose)
    else: # it is mx or pt
        return run_opennlp(training_events_file.name, eval_events_file.name,
                           model_type, smoothing_parameter, verbose)


def run_opennlp (training_filename, eval_filename, model_type, smoothing_parameter, verbose):
    
    # Used to hide verbose output from OpenNLP Maxent
    redirect_output_string = "" if verbose else ' > /dev/null 2> /dev/null '

    model_option = " -perceptron " if model_type == "pt" else ""

    # Train a model
    model_file = tempfile.NamedTemporaryFile(suffix="gz")
    os.system('classify train -cutoff 1 -maxit 1000 -sigma ' + str(smoothing_parameter) + ' ' \
              + model_option + training_filename + ' ' + model_file.name + redirect_output_string)

    # Use the model to predict labels for the evaluation set
    predictions_file = tempfile.NamedTemporaryFile()
    os.system('classify apply ' + model_file.name + ' ' + eval_filename + ' > ' + predictions_file.name)

    # Pull predictions from outputfile and insert labels back into tweetset
    return [tuple(x.decode().strip().split(' ')[0:2]) for x in predictions_file.readlines()]

def run_naivebayes (training_filename, eval_filename, smoothing_parameter, verbose):
    
    # Used to hide verbose output from OpenNLP Maxent
    verbosity_string = " -v " if verbose else ''

    # Train and use the model to predict labels for the evaluation set
    predictions_file = tempfile.NamedTemporaryFile()
    os.system('./naivebayes.py -l ' + str(smoothing_parameter) + verbosity_string \
              + ' -t ' + training_filename + ' -p ' + eval_filename + ' -o ' + predictions_file.name)

    # Pull predictions from outputfile and insert labels back into tweetset
    return [tuple(x.decode().strip().split(' ')[0:2]) for x in predictions_file.readlines()]

def writeResults (type, accuracy, label_results, output_file):
    output_file.write("----------------------------------------\n")
    output_file.write("%s Evaluation\n" % type)
    output_file.write("----------------------------------------\n")
    output_file.write("\t\t%.2f\tOverall Accuracy\n" % accuracy)
    
    output_file.write("----------------------------------------\n")
    output_file.write("P\tR\tF\n")
    (avg_precision, avg_recall) = (0,0)
    for (p,r,f,l) in label_results:
        output_file.write("%.2f\t%.2f\t%.2f\t%s\n" % (p,r,f,l))
        avg_precision += p
        avg_recall    += r
        
    num_labels = len(label_results)
    avg_precision /= num_labels
    avg_recall    /= num_labels
    avg_fscore     = 2*avg_precision*avg_recall/(avg_precision+avg_recall)

    output_file.write("........................................\n")
    output_file.write("%.2f\t%.2f\t%.2f\t%s\n" % (avg_precision, avg_recall, avg_fscore, "Average"))
    output_file.write("----------------------------------------\n")
