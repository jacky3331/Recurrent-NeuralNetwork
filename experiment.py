################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime

from caption_utils import bleu1, bleu4
from constants import ROOT_STATS_DIR
from dataset_factory import get_datasets
from file_utils import *
from model_factory import get_model

################################################################################
# Our imports:
import torch.nn as nn
from pycocotools.coco import COCO
import nltk
################################################################################


# Class to encapsulate a neural experiment.
# The boilerplate code to setup the experiment, log stats, checkpoints and plotting have been provided to you.
# You only need to implement the main training logic of your experiment and implement train, val and test methods.
# You are free to modify or restructure the code as per your convenience.
class Experiment(object):
    def __init__(self, name):
        config_data = read_file_in_dir('./', name + '.json')
        self.config_data = config_data
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)

        self.__name = config_data['experiment_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)

        # Load Datasets
        self.__coco_test, self.__vocab, self.__train_loader, self.__val_loader, self.__test_loader = get_datasets(
            config_data)

        # Setup Experiment
        self.__generation_config = config_data['generation']
        self.__epochs = config_data['experiment']['num_epochs']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__best_model = None  # Save your best model in this field and use this in test method.
        
        self.__cocoTest = COCO('./data/annotations/captions_val2014.json')

#>>>>    # Init Model
# self.__model = get_model(config_data, self.__vocab)
        self.__model, self.decoder = get_model(config_data, self.__vocab)

        # Change to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__model.to(device)
        self.decoder.to(device)
    
        # TODO: Set these Criterion and Optimizers Correctly
        params = list(self.decoder.parameters()) + list(self.__model.embed.parameters())
        self.__criterion = nn.CrossEntropyLoss()
        self.__optimizer = torch.optim.Adam(params=params, lr=1e-4)

        self.__init_model()

        # Load Experiment Data if available

################################################################################
        # uncomment later
        self.__load_experiment()
################################################################################

    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.__experiment_dir):
            self.__training_losses = read_file_in_dir(self.__experiment_dir, 'training_losses.txt')
            self.__val_losses = read_file_in_dir(self.__experiment_dir, 'val_losses.txt')
            self.__current_epoch = len(self.__training_losses)

            state_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_model.pt'))
            self.__model.load_state_dict(state_dict['model'])
            self.__optimizer.load_state_dict(state_dict['optimizer'])
            
            self.decoder.load_state_dict(state_dict['decoder'])
            device = torch.device("cuda")
            self.__model.to(device)
            self.__model.eval()

        else:
            os.makedirs(self.__experiment_dir)

    def __init_model(self):
        if torch.cuda.is_available():
            self.__model = self.__model.cuda().float()
            self.__criterion = self.__criterion.cuda()

    # Main method to run your experiment. Should be self-explanatory.
    def run(self):
        start_epoch = self.__current_epoch
        for epoch in range(start_epoch, self.__epochs):  # loop over the dataset multiple times
            start_time = datetime.now()
            self.__current_epoch = epoch
            train_loss = self.__train()
            val_loss = self.__val()
            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model()

    # TODO: Perform one training iteration on the whole dataset and return loss value
    def __train(self):
        self.__model.train()
        training_loss = 0
        
        # Additional code:
        counter = 0
        running_loss = 0

        for i, (images, captions, _) in enumerate(self.__train_loader):
            counter += 1
            images, captions = images.cuda(), captions.cuda()
            
            # zero the parameter gradients
#             self.__optimizer.zero_grad()
            self.__model.zero_grad()
            self.decoder.zero_grad()
            
            # Can we set encoder to self.model and just wrap the decoder around it?
            
            # Pass the inputs through the CNN-RNN model.
            features = self.__model(images)
            outputs = self.decoder(features, captions)
            
            # Calculate the loss.
            loss = self.__criterion(outputs.view(-1, self.__vocab.__len__()), captions.view(-1))
            loss.backward()
            self.__optimizer.step()
            
            # collect loss
            running_loss += loss.item()
        
        # Avg. the loss accumulated across entire data set
        training_loss = running_loss/counter

        return training_loss

    # TODO: Perform one Pass on the validation set and return loss value. You may also update your best model here.
    def __val(self):
        self.__save_model()
        self.__model.eval()
        val_loss = 0
        
        
        # Additional code:
        counter = 0
        running_loss = 0

        with torch.no_grad():
            for i, (images, captions, _) in enumerate(self.__val_loader):
                counter += 1
                images, captions = images.cuda(), captions.cuda()
                
                # Pass the inputs through the CNN-RNN model.
                features = self.__model(images)
                outputs = self.decoder(features, captions)
                
                # Calculate and collect the loss.
                loss = self.__criterion(outputs.view(-1, self.__vocab.__len__()), captions.view(-1))
                running_loss += loss.item()
            
            # Avg. the loss accumulated across entire data set
            val_loss = running_loss/counter

        return val_loss

    # TODO: Implement your test function here. Generate sample captions and evaluate loss and
    #  bleu scores using the best model. Use utility functions provided to you in caption_utils.
    #  Note than you'll need image_ids and COCO object in this case to fetch all captions to generate bleu scores.
    def test(self):
        self.__model.eval()
        test_loss = 0
        bleu1_ = 0
        bleu4_ = 0
        coco_object = self.__coco_test
        counter = 0
        max_len = self.config_data['generation']['max_length']
        batch_size = self.config_data['dataset']['batch_size']

        with torch.no_grad():
            for i, (images, captions, img_ids) in enumerate(self.__test_loader):
                counter += 1
                # Put images, captions into the gpu
                images, captions = images.cuda(), captions.cuda()
                
                # Send images through the encoder
                features = self.__model(images).unsqueeze(1)
                
                # Separate part which calculates the loss for the test set
                temp_features = self.__model(images)
                temp_outputs = self.decoder(temp_features, captions)
                test_loss += self.__criterion(temp_outputs.view(-1, self.__vocab.__len__()), captions.view(-1)).item()
                
                # Send the features through the decoder
                if self.config_data['generation']['deterministic'] == True:
                    outputs = self.decoder.generate_captions_deterministic(self.config_data, features)
                else:
                    outputs = self.decoder.generate_captions_stocastic(self.config_data, features)
                
                # Overlay each word stack to the corresponding image
                outputs = np.stack(outputs, axis = 1)
     
                # Converting word ids to words
                predicted_caption = []
                for j in range(len(outputs)):
                    temp_caption = []
                    for k in range(max_len):
                        vocab_id = outputs[j][k]
                        word = self.__vocab.idx2word[vocab_id]
                        if word == '<end>':
                            break
                        if word != '<start>':
                            temp_caption.append(word)
                            
                    temp_caption = " ".join(temp_caption)
                    predicted_caption.append(temp_caption.split())
                
                # Calculating Bleu scores
                for item in range(len(img_ids)):
                    load_captions = self.__cocoTest.loadAnns(self.__cocoTest.getAnnIds(img_ids[item]))
                    actual_captions = [nltk.word_tokenize(sentence['caption'].lower()) for sentence in load_captions]
                    predicted = predicted_caption[item]
                    
                    bleu1_ += bleu1(actual_captions, predicted)/100
                    bleu4_ += bleu4(actual_captions, predicted)/100
                    
            # Calcuate avg bleu1, bleu4, loss among all the images
            bleu1_ /= (counter * batch_size)
            bleu4_ /= (counter * batch_size)
            test_loss /= counter

        result_str = "Test Performance: Loss: {}, Bleu1: {}, Bleu4: {}".format(test_loss,
                                                                                               bleu1_,
                                                                                               bleu4_)
        self.__log(result_str)

        return test_loss, bleu1, bleu4

    def __save_model(self):
        root_model_path = os.path.join(self.__experiment_dir, 'latest_model.pt')
        model_dict = self.__model.state_dict()
        
        decoder_dict = self.decoder.state_dict()
        
        state_dict = {'model': model_dict, 'decoder': decoder_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

    def __record_stats(self, train_loss, val_loss):
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)

        self.plot_stats()

        write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)

    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self):
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_losses, label="Training Loss")
        plt.plot(x_axis, self.__val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        plt.show()
