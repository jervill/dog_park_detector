#!/usr/bin/env python3
# Get files
## Print number of files to work on
# Check that output directories exist

# Function to crop files
## Within this function, save to the appropriate directory, appending to name

import os
import argparse

LABELS = ['High Activity', 'Low Activity', 'No Activity']

# Only expecting to work with .jpeg images
def image_check(filename):
    return filename[-4:] == 'jpeg'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_version', required=True,
        help='Version of the model. Used as the file name.')
    parser.add_argument('--model', required=True,
        help='Type of model. Values can be: Dog, VB1, VB2')
    args = parser.parse_args()

    cwd = os.getcwd()

    print('Currently at', cwd)
    print('Looking for label directories.')

    try:
        parent_directory = os.listdir('./')
        for label in LABELS:
            parent_directory.index(label)
    except ValueError:
        print('One of the directories was missing.')
        exit()

    filename = '{model}-{version}.txt'.format(model=args.model, version=args.model_version)

    with open(filename, 'w') as file:
        string = ''

        # create the list for each label.
        # Write it to a file.
        for label in LABELS:
            string += '{label}:\n'.format(label=label)
            images = os.listdir('./' + label)
            images = list(filter(image_check, images))

            print('Found', len(images), 'in', label)

            for image in images:
                string += '    {image}'.format(image=image)
                string += '\n'

        file.write(string)


if __name__ == '__main__':
    main()