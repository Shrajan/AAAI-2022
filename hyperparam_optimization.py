import numpy as np
import torch, time, utils, os, nrrd
import torch.nn as nn

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker
import hpbandster.core.result as hpres

import dataloader as DL

import logging
logging.basicConfig(level=logging.WARNING)

from options import Options

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB as BOHB
from hpbandster.examples.commons import MyWorker

import hpbandster.core.result as hpres
import hpbandster.visualization as hpvis

from train import Trainer
import csv
import copy

# Initializing.
opt = Options().parse()

class MyWorker(Worker):
    def __init__(self, *args, sleep_interval=0, results_path=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.sleep_interval = sleep_interval
        self.config_number = 0
        self.results_path = results_path
        self.overall_highest_mean_dsc = 0.0

    def compute(self, config, budget, **kwargs):
        """
        Args:
            config: dictionary containing the sampled configurations by the optimizer
            budget: (float) amount of time/epochs/etc. the model can use to train
        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """

        # Saving hpbandster config in options.
        opt.lr = config["lr"]
        opt.beta1 = config["beta1"]
        opt.dropout_rate = config["dropout_rate"]

        self.config_number += 1

        print("\nStarted worker with configuration:\n",
          "lr = {}, beta1 = {}, dropout rate = {}".format(
            config["lr"], config["beta1"], config["dropout_rate"]
          )
        )


        ####################################################
        #----------------- CREATE TRAINER -----------------#
        ####################################################

        trainer = Trainer(opt=opt, hyperparam_optimization=True)

        # Variable to determine when to store current best results.
        highest_mean_dsc = 0.0
        best_val_metrics = dict()

        history_file = self.results_path + "/config_" + str(self.config_number) + "_history.csv"
        detailed_history_file = self.results_path + "/config_" + str(self.config_number) + "_detailed_history.csv"

        with open(history_file, "w", newline="") as file:
            writer = csv.writer(file, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["epoch", "mean_dsc", "mean_hsd", "mean_icc", "mean_ari"])

        with open(detailed_history_file, "w", newline="") as file:
            writer = csv.writer(file, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["epoch", "iterations", "dsc", "hsd", "icc", "ari"])

        history = {
          "DSC": [],
          "HSD": [],
          "ICC": [],
          "ARI": [],
          "Epoch": [],
        }

        # Loop
        for epoch in range(int(budget)):

            ####################################################
            #------------------ TRAINING STEP -----------------#
            ####################################################

            train_losses, train_dsc = trainer.train_step()

            ####################################################
            #----------------- VALIDATION STEP ----------------#
            ####################################################

            val_losses, val_metrics = trainer.val_step(epoch=epoch, save_freq=opt.save_freq)
            
            # Append values to the history dictionary.
            history["Epoch"].append(epoch+1)
            history["DSC"].append(np.mean(val_metrics["DSC"]))
            history["HSD"].append(np.mean(val_metrics["HSD"]))
            history["ICC"].append(np.mean(val_metrics["ICC"]))
            history["ARI"].append(np.mean(val_metrics["ARI"]))
            
            current_mean_dsc = np.mean(val_metrics["DSC"])
            if current_mean_dsc > highest_mean_dsc:
                # Not taking a chance with Python referencing.
                highest_mean_dsc = copy.copy(current_mean_dsc)
                best_val_metrics = copy.deepcopy(val_metrics)

                if highest_mean_dsc > self.overall_highest_mean_dsc and epoch > int(budget/2.5):
                    self.overall_highest_mean_dsc =copy.copy(highest_mean_dsc)        

                    # Save the best model of all the configurations so far.             
                    torch.save({
                    "epoch": epoch+1,
                    "mean_val_DSC": self.overall_highest_mean_dsc,
                    "model_state_dict": trainer.model.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                    "normalize": opt.normalize,
                    "patch_shape": opt.patch_shape,
                    "n_kernels": opt.n_kernels,
                    "no_clip": opt.no_clip,
                    "input_channels": opt.input_channels,
                    "output_channels": opt.output_channels,
                    "no_shuffle": opt.no_shuffle,
                    "voxel_spacing": opt.voxel_spacing,
                    "model_name": opt.model_name
                    }, os.path.join(self.results_path, "overall_best_net.sausage"))


            with open(detailed_history_file, "a", newline="") as file:
                writer = csv.writer(file, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
                for iters in range(len(val_metrics["DSC"])):
                  writer.writerow([epoch+1, iters+1, val_metrics["DSC"][iters], val_metrics["HSD"][iters], \
                                  val_metrics["ICC"][iters], val_metrics["ARI"][iters] ])

            print("Finished {}/{} epochs.".format(epoch + 1, int(budget)), end="\r", flush=True)

        meanDSC, medDSC = np.mean(best_val_metrics["DSC"]), np.median(best_val_metrics["DSC"]) # Dice-soerensen coefficient.
        print("Best mean val DSC = {}, best median val DSC = {}".format(meanDSC, medDSC))

        with open(history_file, "a", newline="") as file:
            writer = csv.writer(file, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for count in range(len(history["Epoch"])):
              writer.writerow([history["Epoch"][count], history["DSC"][count], history["HSD"][count], \
                              history["ICC"][count], history["ARI"][count] ])


        accuracy = 1 - meanDSC # Using the inverse, because hpbandster minimizes.

        return {
            "loss": float(accuracy),  # Mean validation accuracy (1 - mean(DSC))
            "info": {
              'mean DSC': meanDSC,
              'median DSC': medDSC,
              'DSC': best_val_metrics['DSC'],
              'HSD': best_val_metrics['HSD'],
              'ICC': best_val_metrics['ICC'],
              'ARI': best_val_metrics['ARI']
            }  # A bunch of metrics.
        }

    @staticmethod
    def get_configspace():
        config_space = CS.ConfigurationSpace()

        # Defining hyperparameters that shall be optimized.
        lr = CSH.UniformFloatHyperparameter('lr', lower=opt.lr_min, upper=opt.lr_max, default_value='2e-4', log=True)
        beta1 = CSH.UniformFloatHyperparameter('beta1', lower=opt.beta1_min, upper=opt.beta1_max, default_value=0.5, log=False)
        dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate', lower=opt.dropout_rate_min, upper=opt.dropout_rate_max, default_value=0.25, log=False)
        
        config_space.add_hyperparameters([lr, beta1, dropout_rate])
        return config_space



def main():
    # Start a Nameserver.
    host = hpns.nic_name_to_host("lo")
    NS = hpns.NameServer(run_id=opt.run_id, host=host, port=opt.port)
    ns_host, ns_port = NS.start()

    # Start a worker.
    w = MyWorker(sleep_interval = 0, results_path=opt.results_path, host=host, nameserver=ns_host, nameserver_port=ns_port, run_id=opt.run_id)
    w.run(background=True)

    # Create a result logger for live result logging.
    result_logger = hpres.json_result_logger(directory=opt.results_path, overwrite=False)

    # Continue search from the last saved point.
    if opt.previous_search_path == "None":

        # Run an optimizer.
        bohb = BOHB(  configspace = w.get_configspace(),
                  run_id = opt.run_id,
                  host=host,
                  nameserver=ns_host,
                  nameserver_port=ns_port, result_logger=result_logger,
                  min_budget=opt.min_budget, max_budget=opt.max_budget,
              )

    else:
        if opt.previous_search_path == opt.results_path:
            raise Exception("Please use different path to store new results.")

        previous_run = hpres.logged_results_to_HBS_result(opt.previous_search_path)

        # Run an optimizer.
        bohb = BOHB(  configspace = w.get_configspace(),
                  run_id = opt.run_id,
                  host=host,
                  nameserver=ns_host,
                  nameserver_port=ns_port, result_logger=result_logger,
                  min_budget=opt.min_budget, max_budget=opt.max_budget,
                  previous_result = previous_run
               )
    res = bohb.run(n_iterations=opt.n_iterations)

    # Shut down worker and nameserver.
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()

    # Get all executed runs.
    all_runs = res.get_all_runs()

    # Analysis.
    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()

    print('Best found configuration:', id2config[incumbent]['config'])
    print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
    print('A total of %i runs where executed.' % len(res.get_all_runs()))
    print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in res.get_all_runs()])/opt.max_budget))

    final_msg = ""
    final_msg += 'Best found configuration:' + str(id2config[incumbent]['config']) + "\n"
    final_msg += 'A total of %i unique configurations where sampled.' % len(id2config.keys()) + "\n"
    final_msg += 'A total of %i runs where executed.' % len(res.get_all_runs()) + "\n"
    final_msg += 'Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in res.get_all_runs()])/opt.max_budget) +"\n"

    with open(opt.results_path + "/summary.txt", "w") as file_object:
      file_object.write(final_msg)
    
    # Visualization of runs.
    lcs = res.get_learning_curves()

    hpvis.interactive_HBS_plot(lcs, tool_tip_strings=hpvis.default_tool_tips(res, lcs))

if __name__ == "__main__":
    main()
