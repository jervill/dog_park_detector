# Get files
## Print number of files to work on
# Check that output directories exist

# Function to crop files
## Within this function, save to the appropriate directory, appending to name

from PIL import Image
import os

cwd = os.getcwd()
print 'Currently at ' + cwd
print 'Looking for destination directories "../Dog", "../VB1", and "../VB2".'

try:
    parent_directory = os.listdir('../')
    parent_directory.index('Dog')
    parent_directory.index('VB1')
    parent_directory.index('VB2')
except ValueError:
    print 'One of the directories was missing.'
    exit()

# Get files to work with
files = os.listdir('./')

# Only expecting to work with .jpeg images
def image_check(filename):
    return filename[-4:] == 'jpeg'

files = filter(image_check, files)

# Crop the files
def crop(file):
    with Image.open(file) as input:
        vb1 = input.crop(box=(0,240,170,400))
        vb2 = input.crop(box=(188, 240, 408,400))
        dog = input.crop(box=(559, 228, 820,388))

        vb1_name = '../VB1/vb1-' + file
        vb2_name = '../VB2/vb2-' + file
        dog_name = '../Dog/dog-' + file

        vb1.save(vb1_name)
        vb2.save(vb2_name)
        dog.save(dog_name)

        print 'Volleyball court 1 saved to ' + vb1_name
        print 'Volleyball court 2 saved to ' + vb2_name
        print 'Dog park saved to ' + dog_name



for file in files:
    print 'Cropping ' + file
    crop(file)

