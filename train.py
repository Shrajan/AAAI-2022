import dataloader as DL
import torch, os, nrrd, time
from options import Options
import utils
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import csv
import models

class Trainer():
  def __init__(self, opt=None, hyperparam_optimization=False):

        print("Initializing...")
        self.start_time = time.time()

        # Initializing.
        if opt is None:
            self.opt = Options().parse()
        else:
            self.opt = opt

        # Location to save the current training experiment.
        self.results_path = self.opt.results_path

        # Values to save best model.
        self.highestDSC = 0.0

        self.resume_train = self.opt.resume_train

        if not self.resume_train and hyperparam_optimization==0:

                # Saving validation predictions if needed every nth epoch.
                if self.opt.save_freq > 0:
                    self.prediction_path = os.path.join(self.results_path,'predictions')
                    utils.create_folder(self.prediction_path)

                # Store epochs wise results.
                with open(self.results_path + "/history.csv", "w", newline="") as file:
                    writer = csv.writer(file, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow(["epoch","train_loss", "train_dsc", "train_hsd", "train_icc", "train_ari", "train_assd", \
                    "val_loss", "val_dsc", "val_hsd", "val_icc", "val_ari", "val_assd"])

                # Store iteration wise resuls of the training samples.
                with open(self.results_path + "/training_iteration_results.csv", "w", newline="") as file:
                    writer = csv.writer(file, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow(["epoch", "iterations", "loss", "dsc", "hsd", "icc", "ari", "assd"])
                
                # Store iteration wise resuls of the validation samples.
                with open(self.results_path + "/validation_iteration_results.csv", "w", newline="") as file:
                    writer = csv.writer(file, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow(["epoch", "iterations", "loss", "dsc", "hsd", "icc", "ari", "assd"])

        ####################################################
        #------------------ DATA LOADING ------------------#
        ####################################################

        self.train_dataset, self.val_dataset = DL.Dataset(self.opt, training=True), DL.Dataset(self.opt, training=False)
        self.trainloader = DataLoader(
            self.train_dataset,
            batch_size=self.opt.n_batches,
            shuffle=True,
            num_workers=self.opt.n_workers,
            pin_memory=True,
            drop_last=True
            )
        self.valloader = DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.opt.n_workers,
            pin_memory=True
            )

        ####################################################
        #----------------- MODEL CREATION -----------------#
        ####################################################

        device_id = 'cuda:' + str(self.opt.device_id)
        self.device = torch.device(device_id if torch.cuda.is_available() else 'cpu')

        self.model = models.get_model(opt=self.opt)

        # Further training
        if self.resume_train:
            checkpoint = torch.load(os.path.join(self.results_path,'best_net.sausage'), map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"]) # Load the best weights
        self.model.to(self.device)

        ####################################################
        #-------------- TRAINING PARAMETERS ---------------#
        ####################################################

        # Set optimizer
        self.optimizer = utils.set_optimizer(opt=self.opt, model_params=self.model.parameters())

        # Learning rate scheduler.
        t_max = self.opt.training_epochs if not hyperparam_optimization else self.opt.max_budget
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t_max, eta_min=0.000001)

        # Initialize loss functions.
        self.loss_func = utils.set_loss_fn(opt=self.opt)

  ###############################################################
  #------------------------- TRAINING --------------------------#
  ###############################################################
  def train_step(self):
      """
      Train the model on all training samples.
      Returns:
          train_losses: list of all computed training losses
          train_dsc: list of all computed training DSCs
      """
      # List to store loss values and metrics generated per single batch.
      train_losses  = []
      train_metrics = {
          "DSC": [],
          "HSD": [],
          "ICC": [],
          "ARI": [],
          "ASSD": [],
      }

      # Perform training by setting mode = Train
      self.model.train()

      # Consider a single batch at a given time from the complete dataset.
      for idx, (p_id, data, label, p_idx) in enumerate(self.trainloader):

          self.optimizer.zero_grad()                                                            # Clears old gradients from the pevious step.
          data, label = data.to(self.device), label.to(self.device)                             # Pass data and label to the device.
          out_M = self.model(data)                                                              # Feed the data into the model.
          loss_M = self.loss_func(out_M, label)                                                 # Computes the loss of prediction with respect to the label.
          train_losses.append(loss_M)                                                           # Save the batch-wise training loss.
          loss_M.backward()                                                                     # Computes the derivative of the loss w.r.t. the parameters (or anything requiring gradients) using backpropagation.
          self.optimizer.step()                                                                 # Use the optimizer to change the paramters based on their respective gradients.
          train_metrics = utils.compute_metrices(torch.sigmoid(out_M), label, train_metrics, opt=self.opt)    # Calculate the performance metrices and save it in the dictionary.

      return train_losses, train_metrics

  ###############################################################
  #------------------------- VALIDATION ------------------------#
  ###############################################################
  def val_step(self, epoch, save_freq=0):
      """
      Validating the trained model on all validation samples.
      Returns:
          val_losses: list of all computed validation losses
          val_dsc: list of all computed validation DSCs
      """

      # List to store loss values and metrics generated per single batch.
      val_losses = []
      val_metrics = {
          "DSC": [],
          "HSD": [],
          "ICC": [],
          "ARI": [],
          "ASSD": [],
      }

      # Perform validation by setting mode = Eval
      self.model.eval()

      # Make sure that the gradients will not be altered or calculated.
      with torch.no_grad():
          self.val_dataset.shuffle_patch_choice()

          # Consider a single batch at a given time from the complete dataset.
          for idx, (p_id, data, label, p_idx) in enumerate(self.valloader):

              data, label = data.to(self.device), label.to(self.device)                         # Pass data and label to the device.
              out_M = self.model(data)                                                          # Feed the data into the model.
              loss_M = self.loss_func(out_M, label)                                             # Calculate the loss function of the model prediction to the labelled ground truth.
              val_losses.append(loss_M.detach())                                                # Save the batch-wise validation loss.
              val_metrics = utils.compute_metrices(torch.sigmoid(out_M), label, val_metrics, opt=self.opt)    # Calculate the performance metrices and save it in the dictionary.

              # Save every nth epoch determined by save_frequency.
              if save_freq > 0:
                if epoch % save_freq == 0:
                    out_M = torch.sigmoid(out_M)
                    nrrd.write(self.prediction_path + "/" + p_id[0] + "_epoch_" + str(epoch) + "_prediction.nrrd", out_M.squeeze(0).squeeze(0).cpu().detach().numpy())
                    nrrd.write(self.prediction_path + "/" + p_id[0] + "_epoch_" + str(epoch) + "_val_ct_patch.nrrd", data.squeeze(0).squeeze(0).cpu().detach().numpy())
                    nrrd.write(self.prediction_path + "/" + p_id[0] + "_epoch_" + str(epoch) + "_val_gt_patch.nrrd", label.squeeze(0).squeeze(0).cpu().detach().numpy())
        
      self.lr_scheduler.step()
      
      return val_losses, val_metrics

  """
  This method trains the network.
  """
  def train(self):

      for epoch in range(self.opt.starting_epoch, self.opt.training_epochs):

          # Print information
          print("Epoch {}/{}".format(epoch+1, self.opt.training_epochs))
          epoch_start_time = time.time()

          # Training step
          train_losses, train_metrics = self.train_step()

          # Validation step
          val_losses, val_metrics = self.val_step(epoch, save_freq= self.opt.save_freq)
          
          ###############################################################
          #---------------------- SAVING RESULTS -----------------------#
          ###############################################################

          # Detailed training iteration results.
          with open(self.results_path + "/training_iteration_results.csv", "a", newline="") as file:
              writer = csv.writer(file, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
              for iters in range(len(train_losses)):
                  writer.writerow([ epoch+1, iters+1, train_losses[iters].item(), train_metrics["DSC"][iters], train_metrics["HSD"][iters], \
                                    train_metrics["ICC"][iters], train_metrics["ARI"][iters], train_metrics["ASSD"][iters]  ])
          
          # Detailed validation iteration results.
          with open(self.results_path + "/validation_iteration_results.csv", "a", newline="") as file:
              writer = csv.writer(file, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
              for iters in range(len(val_losses)):
                  writer.writerow([ epoch+1, iters+1, val_losses[iters].item(), val_metrics["DSC"][iters], val_metrics["HSD"][iters], \
                                    val_metrics["ICC"][iters], val_metrics["ARI"][iters], val_metrics["ASSD"][iters]  ])

          result = dict()

          # Training results
          result['train_loss'] = torch.stack(train_losses).mean().item()
          result['train_dsc'] = np.mean(train_metrics["DSC"])
          result['train_hsd'] = np.mean(train_metrics["HSD"])
          result['train_icc'] = np.mean(train_metrics["ICC"])
          result['train_ari'] = np.mean(train_metrics["ARI"]) 
          result['train_assd'] = np.mean(train_metrics["ASSD"])         

          # Validation results          
          result['val_loss'] = torch.stack(val_losses).mean().item()
          result['val_dsc'] = np.mean(val_metrics["DSC"])
          result['val_hsd'] = np.mean(val_metrics["HSD"])
          result['val_icc'] = np.mean(val_metrics["ICC"])
          result['val_ari'] = np.mean(val_metrics["ARI"])  
          result['val_assd'] = np.mean(val_metrics["ASSD"])    

          with open(self.results_path + "/history.csv", "a", newline="") as file:
              writer = csv.writer(file, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
              writer.writerow([ epoch+1, result["train_loss"], result["train_dsc"], result["train_hsd"], result["train_icc"], result["train_ari"], result["train_assd"],\
                                    result["val_loss"], result["val_dsc"], result["val_hsd"], result["val_icc"], result["val_ari"], result["val_assd"] ])
          
          # Save the model that has the highest mean validation DSC.
          currentDSC = result['val_dsc']
          if currentDSC > self.highestDSC:
              self.highestDSC = currentDSC
              bestEpoch = epoch+1
              torch.save({
              "epoch": bestEpoch,
              "mean_val_DSC": self.highestDSC,
              "model_state_dict": self.model.state_dict(),
              "optimizer_state_dict": self.optimizer.state_dict(),
              "normalize": self.opt.normalize,
              "patch_shape": self.opt.patch_shape,
              "n_kernels": self.opt.n_kernels,
              "no_clip": self.opt.no_clip,
              "voxel_spacing": self.train_dataset.voxel_spacing,
              "input_channels": self.opt.input_channels,
              "output_channels": self.opt.output_channels,
              "no_shuffle": self.opt.no_shuffle,
              "model_name": self.opt.model_name
              }, os.path.join(self.results_path, "best_net.sausage"))
          
          # Calculate the time required for each epoch.
          epoch_end_time = time.time()
          time_per_epoch = epoch_end_time - epoch_start_time

          # Print the results after each epoch.
          print("Finished epoch {} in {}s with results - train_loss: {:.4f}, train_dsc: {:.4f}, val_loss: {:.4f}, val_dsc: {:.4f} \n".format(
          epoch+1, int(time_per_epoch), result['train_loss'], result['train_dsc'], result['val_loss'], result['val_dsc']))

      # Save the complete model after training.
      torch.save(self.model,os.path.join(self.results_path, 'complete_model.pt'))
      
      # Calculate the time required for full training.
      self.end_time = time.time()
      print("Completed training and validation in {}s".format(self.end_time - self.start_time))

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
