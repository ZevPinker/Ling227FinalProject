#!/usr/bin/env python3

#############################################################################
# Copyright 2011 Jason Baldridge
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#############################################################################

import sys
from porter_stemmer import PorterStemmer
from BitVector import BitVector
from classify_util import *

#############################################################################
# Set up the options
parser = get_feature_extractor_option_parser()
parser.add_option("-b", "--bitstrings", dest="bitstrings",
                  help="read bit vectors from FILE", metavar="FILE")

(options, args) = parser.parse_args()
check_mandatory_options(parser, options, ['input'])

#############################################################################
# Use the options to set the input and output files appropriately
input_file = open(options.input)
output_file = sys.stdout
if options.out != None:
    output_file = file(options.out, "w")

verbose = options.verbose
errmsg("Showing verbose output.", verbose)

# Load the bitstrings if the option has been given
bitstrings = {}
use_bitstrings = False
if options.bitstrings != None:
    use_bitstrings = True
    for line in open(options.bitstrings).readlines():
        (word, bits_for_word) = line.strip().split('\t')
        bitstrings[word] = BitVector(bitstring=bits_for_word)

# Create the stemmer for later use.
stemmer = PorterStemmer()

#############################################################################
# Extract features
for line in input_file.readlines():
    (id, verb, noun, prep, prep_obj, label) = \
        [protect_meta_characters(x) for x in line.strip().split()]

    features = []

    # Add the four basic features
    features.append("verb="+verb)
    features.append("noun="+noun)
    features.append("prep="+prep)
    features.append("prep_obj="+prep_obj)

    # Add additional features. An example has been given for stem
    # features, n-gram features, and bitstring features. It's up to
    # you to come up with other features like those, plus any new
    # features you might think of!
    if options.extended_features:

        # ADD CODE: Stem features
        features.append("verb_stem="+stemmer.stem_token(verb))

        # ADD CODE: N-gram features
        features.append("verb+noun="+verb+"+"+noun)

        # ADD CODE: bitstring features
        # Use the bitstrings to create features. See the paper for a
        # description:
        #
        #     http://aclweb.org/anthology-new/H/H94/H94-1048.pdf
        #
        # These are more advanced than the others, but are very
        # informative.
        if use_bitstrings:

            if noun in bitstrings:
                noun_bv = bitstrings.get(noun)
                features.append("noun_bit1="+str(noun_bv[1]))

        # ADD CODE: Other features.

    # Add the label
    features.append(label)

    # Join and print the features
    output_file.write(','.join(features) + "\n")
