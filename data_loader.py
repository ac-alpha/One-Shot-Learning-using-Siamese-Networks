import torch
import torchvision
import torchvision.transforms as transforms
import glob
import math
import random
from PIL import Image
import time
import numpy as np

def getTrainImages():

    #Image from only the first 12 drawers is taken during train time
    image_path = "data/omniglot-py/images_background/*/character*/*"
    images = []
    for x in range(1,13):
        image_path1=image_path
        if x<10:
            image_path1+="0"
        image_path1+=str(x)
        image_path1+=".png"
        images += glob.glob(image_path1) 
    print(len(images)) # = 11568
    # print(images[0]) # prints the image name
    return images
  
def getValImages():

    #Image from next 4 drawers is taken during val time
    image_path = "data/omniglot-py/images_background/*/character*/*"
    images = []
    for x in range(13,17):
        image_path1=image_path
        if x<10:
            image_path1+="0"
        image_path1+=str(x)
        image_path1+=".png"
        images += glob.glob(image_path1) 
    print(len(images)) # = 11568
    # print(images[0]) # prints the image name
    return images

def getRandomPairedIndices(no, start, end):
    count = 0
    pairs = []
    pair = []
    while (count < no):
        pair = []
        x, y = random.randint(start, end), random.randint(start, end)
        pair.append(x)
        pair.append(y)
        if x==y:
            continue
        if x>y:
            x, y = y, x
        if pair in pairs:
            continue
        pairs.append(pair)
        count+=1;
    return pairs

def getTrainExamples(n = 30000):

    all_images = getTrainImages()
    number_of_char = 964

    # For same characters
    n_same = n / 2
    same_images_per_char = math.ceil(n_same / number_of_char)
    n_same = same_images_per_char * number_of_char
    same_character_examples = []
    char_index = 0
    pair = []
    print_every = 1000
    trans = transforms.ToTensor()
#     We need to get n / number_of_char pairs from a single character (12 drawers)
#     So we will get n / number_of_char random pairs of indexes from 0-11
    for i in range(0, number_of_char):
        indices = getRandomPairedIndices(same_images_per_char, 0, 11)
        for index in indices:
            pair = []
            print(i)
            img1 = Image.open(all_images[index[0]*number_of_char + i])
            img1 = trans(img1).numpy()
            img2 = Image.open(all_images[index[1]*number_of_char + i])
            img2 = trans(img2).numpy()
            pair.append(img1)
            pair.append(img2)
            same_character_examples.append(pair)

#    For different characters
    n_diff = n_same
    count = 0
    i1,i2 = 0,0
    diff_character_examples = []
    while(count<n_diff):
        pair = []
        print("count %d"%(count))
        i1 = random.randint(0,len(all_images)-1)
        i2 = random.randint(0,len(all_images)-1)
        img1 = Image.open(all_images[i1])
        img1 = trans(img1).numpy()
        img2 = Image.open(all_images[i2])
        img2 = trans(img2).numpy()
        pair.append(img1)
        pair.append(img2)
        diff_character_examples.append(pair)
        count+=1

    random.shuffle(same_character_examples)
    random.shuffle(diff_character_examples)
    
    return same_character_examples,diff_character_examples

def getValExamples(n = 1000):

    all_images = getValImages()
    number_of_char = 964

    # For same characters
    n_same = n / 2
    same_images_per_char = math.ceil(n_same / number_of_char)
    n_same = same_images_per_char * number_of_char
    same_character_examples = []
    char_index = 0
    pair = []
    trans = transforms.ToTensor()
#     We need to get n / number_of_char pairs from a single character (4 drawers)
#     So we will get n / number_of_char random pairs of indexes from 0-11
    for i in range(0, number_of_char):
        indices = getRandomPairedIndices(same_images_per_char, 0, 3)
        for index in indices:
            pair = []
            print(i)
            img1 = Image.open(all_images[index[0]*number_of_char + i])
            img1 = trans(img1).numpy()
            img2 = Image.open(all_images[index[1]*number_of_char + i])
            img2 = trans(img2).numpy()
            pair.append(img1)
            pair.append(img2)
            same_character_examples.append(pair)

#    For different characters
    n_diff = n_same
    count = 0
    i1,i2 = 0,0
    diff_character_examples = []
    while(count<n_diff):
        pair = []
        print("count %d"%(count))
        i1 = random.randint(0,len(all_images)-1)
        i2 = random.randint(0,len(all_images)-1)
        img1 = Image.open(all_images[i1])
        img1 = trans(img1).numpy()
        img2 = Image.open(all_images[i2])
        img2 = trans(img2).numpy()
        pair.append(img1)
        pair.append(img2)
        diff_character_examples.append(pair)
        count+=1

    random.shuffle(same_character_examples)
    random.shuffle(diff_character_examples)
    
    return same_character_examples,diff_character_examples

def getTrainBatches(n = 30000, batch_size = 128):
    same_character_examples, diff_character_examples = getTrainExamples(n)
    train_batches = []
    current_batch = []
    current_batch_size = 0
    examples_covered = 0
    print_every = 1000
    while examples_covered < n:
        if examples_covered % 2 == 0:
            current_batch.append(same_character_examples.pop())
        else:
            current_batch.append(diff_character_examples.pop())
        current_batch_size+=1
        if current_batch_size == batch_size:
            train_batches.append(torch.tensor(current_batch))
            current_batch = []
            current_batch_size = 0
        if(examples_covered % print_every == 0):
          print("Loading data.... Complete: %d examples"%(examples_covered))
        examples_covered+=1
    if current_batch_size!=0:
        train_batches.append(torch.tensor(current_batch))

    return train_batches

def getValBatch(n = 1000):
    same_character_examples, diff_character_examples = getValExamples(n)
    val_batch = []
    examples_covered = 0
    while examples_covered < n:
        if examples_covered % 2 == 0:
            val_batch.append(same_character_examples.pop())
        else:
            val_batch.append(diff_character_examples.pop())
        examples_covered+=1
          
    return torch.tensor(val_batch)


    


