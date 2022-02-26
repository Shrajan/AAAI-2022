import numpy as np
import torch, cc3d
import os, time, csv, argparse
import nrrd
import utils
from datetime import datetime
import SimpleITK as sitk
from torch.utils.data import DataLoader
from models import get_model
from os import listdir
from os.path import isfile, join
from natsort import natsorted
from collections import OrderedDict


class TestDataset(torch.utils.data.Dataset):
    """
    Read data.nrrd files from data_root, segment with generator
    and store segmentations.

    !ATTENTION!
    Make sure to use the same options as for the training.
    Otherwise segmentation quality may be very poor.
    """
    def __init__(self, opt):
        self.opt = opt
        self.data_root = opt.data_root
        self.voxel_spacing = self.opt.voxel_spacing
        self.patient_ids, self.data_paths, self.label_paths = self.read_data_paths()

        # Split data into training and validation set.
        self.data_idx = np.arange(len(self.patient_ids))
        np.random.seed(2021)
        if not opt.no_shuffle == True:
          self.data_idx = np.random.permutation(self.data_idx)

        if opt.dataset_contains == "train_and_test":
          self.train_val_size = int(0.8*len(self.patient_ids))
          self.test_size = len(self.patient_ids) - self.train_val_size

          if opt.fold == -1:
            self.train_size = int(0.8*self.train_val_size)
            self.val_size =  self.train_val_size - self.train_size

            self.test_idx = self.data_idx[:self.test_size]
            self.val_idx = self.data_idx[self.test_size:self.test_size+self.val_size]
            self.train_idx = self.data_idx[self.test_size+self.val_size:]
            
          else:
            if opt.fold == 0:
              self.test_idx = self.data_idx[:self.test_size]
              self.train_val_idx = self.data_idx[self.test_size:]

            elif opt.fold == 1:
              self.test_idx = self.data_idx[self.test_size:self.test_size*2]
              list1, list2 = list(self.data_idx[:self.test_size]), list(self.data_idx[self.test_size*2:])
              self.train_val_idx = list1 + list2

            elif opt.fold == 2:
              self.test_idx = self.data_idx[self.test_size*2:self.test_size*3]
              list1, list2 = list(self.data_idx[:self.test_size*2]), list(self.data_idx[self.test_size*3:])
              self.train_val_idx = list1 + list2
            
            elif opt.fold == 3:
              self.test_idx = self.data_idx[self.test_size*3:self.test_size*4]
              list1, list2 = list(self.data_idx[:self.test_size*3]), list(self.data_idx[self.test_size*4:])
              self.train_val_idx = list1 + list2

            elif opt.fold == 4:
              self.test_idx = self.data_idx[self.test_size*4:]
              self.train_val_idx = list(self.data_idx[:self.test_size*4])
            
            self.train_val_idx = np.random.permutation(self.train_val_idx)
            self.train_idx = self.train_val_idx[:int(0.8*len(self.train_val_idx))]
            self.val_idx = self.train_val_idx[int(0.8*len(self.train_val_idx)):]

        elif opt.dataset_contains == "test":
          self.test_idx = self.data_idx

        self.data_header = []

    def read_data_paths(self):
        """
        Reads data paths.
        """
        listdir()
        data_root = self.opt.data_root

        # Read patient ids.
        patient_ids = [
            f for f in listdir(data_root) if not isfile(join(data_root, f))
        ]
        patient_ids = natsorted(patient_ids)

        # Read data and label paths.
        data_paths = []
        label_paths = []

        for p_id in patient_ids:
            data_paths.append(join(data_root, p_id, "data.nrrd"))
            label_paths.append(join(data_root, p_id, "label.nrrd"))

        return patient_ids, data_paths, label_paths

    def __len__(self):
        return len(self.test_idx)

    def get_header(self, idx):
        return self.data_header[idx]

    def __getitem__(self, idx):
        """
        Read data and label and return them.
        """
        # Get test indices from the list
        idx = self.test_idx[idx]

        # Read data from memory.
        self.data_header.append(nrrd.read_header(self.data_paths[idx]))
        data = sitk.ReadImage(self.data_paths[idx])
        # Transpose, because sitk uses different coordinate system than pynrrd.
        data = sitk.GetArrayFromImage(data).transpose(2, 1, 0)
        data = np.float32(data)

        # Clip data to 5 and 95 percentiles.
        if not self.opt.no_clip == True:
          low, high = np.percentile(data, [0.5, 99.5])
          data = np.clip(data, low, high)

        data = torch.FloatTensor(data)

        label = sitk.ReadImage(self.label_paths[idx])
        # Transpose, because sitk uses different coordinate system than pynrrd.
        label = sitk.GetArrayFromImage(label).transpose(2, 1, 0)
        label = np.uint8(label)
        label = np.where(label != 0, 1, label)
        label = torch.ByteTensor(label)

        out = OrderedDict()
        out["p_id"] = self.patient_ids[idx]
        out["data"] = data
        out["label"] = label

        return out


def get_gaussian_window(shape, sigma=0.5,  dtype=torch.float32):
  gauss_win = torch.zeros(shape, dtype=dtype)
  pi = torch.acos(torch.zeros(1)).item() * 2

  if len(shape) == 3:
    for x in range(shape[0]):
      for y in range(shape[1]):
        for z in range(shape[2]):
          N = (1 / (sigma **3) * ((2 * pi) ** (3/2)))
          gauss_win[x, y, z] = min(1, N * torch.exp(torch.tensor(-((x-shape[0]/2)**2 + (y-shape[1]/2)**2 + (z-shape[2]/2)**2) / 2 * sigma**2, dtype=dtype)))

  return gauss_win

def keep_only_largest_connected_component(seg):
  """
  Takes a binary 3D tensor and removes all contours that
  include less voxels than the largest one.
  """
  # Compute connected components.
  seg = seg.numpy().astype(np.uint8)
  conn_comp = cc3d.connected_components(seg, connectivity=6)

  # Count number of voxels of each component and find largest component.
  unique, counts = np.unique(conn_comp, return_counts=True)

  # Remove largest component, because it is background.
  idx_largest = np.argmax(counts)
  val_largest = unique[idx_largest]

  counts = np.delete(counts, idx_largest)
  unique = np.delete(unique, idx_largest)

  idx_second_largest = np.argmax(counts)
  val_second_largest = unique[idx_second_largest]

  # Remove all smaller components.
  out = np.zeros_like(conn_comp)
  out = np.where(conn_comp == val_second_largest, 1, out)
  return torch.ByteTensor(out.astype(np.uint8))

def get_centroid(seg):
  """
  Input param: 3D binary array.
  Computes the centroid by taking the mittle of the
  outer boundaries. Returns center coordinates as tuple
  (x, y, z)
  """
  nonzeros = torch.nonzero(seg)
  maxX, maxY, maxZ = nonzeros[:, 0].max().item(), nonzeros[:, 1].max().item(), nonzeros[:, 2].max().item()
  minX, minY, minZ = nonzeros[:, 0].min().item(), nonzeros[:, 1].min().item(), nonzeros[:, 2].min().item()
  return (maxX - ((maxX - minX) // 2), maxY - ((maxY - minY) // 2), maxZ - ((maxZ - minZ) // 2))

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_contains', type=str, default='train_and_test', required=True, choices=["train_and_test", "test"], help='Mention whether the dataset contains test samples only or both training and test samples. Combined dataset is used for cross-validation.')
    parser.add_argument("--divisor", type=int, default=2, required=False, help="Patch overlap for sliding window. Overlap is 1/divisor. More = slower computation time, less = worse performance... More or less...")
    parser.add_argument("--trained_model_path", type=str, default="best_net.sausage", required=True, help="Path to trained model.")
    parser.add_argument("--results_path", type=str, default="results/test/", required=False, help="Path to store the results.")
    parser.add_argument("--data_root", type=str, default="", required=True, help="Path to data.")
    parser.add_argument('--prob_map', action='store_true', required=False, help='If set, probability map will also be stored on memory. This will increase computation times drastically!')
    parser.add_argument('--device_id', type=int, default=0, required=False, help='Use the different GPU(device) numbers available. Example: If two Cuda devices are available, options: 0 or 1')
    parser.add_argument("--model_name", type=str, default="oocs_unet_wmp", required=False, help="Select name of trained model. If available, model name will be read from model state dict, else from this option.")
    parser.add_argument('--input_channels', type=int, default=1, required=False, help='Number of channels in the input data. If available, it will be read from model state dict, else from this option.')
    parser.add_argument('--output_channels', type=int, default=1, required=False, help='Number of channels in the output label. If available, it will be read from model state dict, else from this option.')
    parser.add_argument('--fold', type=int, default=-1, required=False, choices = [-1,0,1,2,3,4], help='Select the fold index for 5-fold crosss validation. If fold == -1 all data in the folder will be used for training and validation.')
    parser.add_argument('--no_shuffle', action='store_true', required=False, help='If set, training and validation data will not be shuffled')
    opt = parser.parse_args()

    # Create results files.
    if not os.path.exists(opt.results_path):
      os.makedirs(opt.results_path)

    with open(join(opt.results_path, "results_detailed.csv"), 'w', newline='') as csvfile:
      writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
      writer.writerow(["Patient", "DSC", "HSD", "ICC", "ARI", "ASSD", "Comp. time (s)", "Comp. time total (s)"])
    
    with open(join(opt.results_path, "results.csv"), 'w', newline='') as csvfile:
      writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
      writer.writerow(["", "DSC", "HSD", "ICC", "ARI", "ASSD", "Comp. time (s)", "Comp. time total (s)"])

    # Load information about trained model from saved model.

    device_id = "cuda:" + str(opt.device_id)
    device = torch.device(device_id if torch.cuda.is_available() else "cpu")
    print("Using device : {}".format(device))

    checkpoint = torch.load(os.path.join(opt.trained_model_path), map_location=device)

    opt.normalize = checkpoint["normalize"]
    opt.patch_shape = checkpoint["patch_shape"]
    opt.n_kernels = checkpoint["n_kernels"]
    opt.no_clip = checkpoint["no_clip"]

    try:
      opt.model_name = checkpoint["model_name"]
    except:
      opt.model_name = opt.model_name
      
    highestDSC = checkpoint["mean_val_DSC"]
    try:
      opt.voxel_spacing = checkpoint["voxel_spacing"]
    except:
      opt.voxel_spacing = (1.099609375,1.099609375,3.0)

    try:
      opt.input_channels = checkpoint["input_channels"]
    except:
      opt.input_channels = opt.input_channels

    try:
      opt.output_channels = checkpoint["output_channels"]
    except:
      opt.output_channels = opt.output_channels

    try:
      opt.no_shuffle = checkpoint["no_shuffle"]
    except:
      opt.no_shuffle = opt.no_shuffle
    opt.dropout_rate = 0.5 # This is just a formality for model declaration.

    # Write options into file.
    with open(join(opt.results_path, "test_options.txt"), "w") as f:
      f.write("opt.normalize = " + str(opt.normalize) + "\n")
      f.write("opt.patch_shape = " + str(opt.patch_shape) + "\n")
      f.write("opt.n_kernels = " + str(opt.n_kernels) + "\n")
      f.write("opt.no_clip = " + str(opt.no_clip) + "\n")
      f.write("opt.voxel_spacing = " + str(opt.voxel_spacing) + "\n")
      f.write("opt.trained_model_path = " + str(opt.trained_model_path) + "\n")
      f.write("opt.divisor = " + str(opt.divisor) + "\n")
      f.write("opt.results_path = " + str(opt.results_path) + "\n")
      f.write("opt.data_root = " + str(opt.data_root) + "\n")
      f.write("opt.model_name = " + str(opt.model_name) + "\n")

    dataset = TestDataset(opt)

    # Load trained model.
    print("Found model that was trained for {} epochs with highest DSC {}, {} normalization, patch shape {} and no_clip = {}. Loading... ".format(checkpoint["epoch"], highestDSC, checkpoint["normalize"], checkpoint["patch_shape"], opt.no_clip), end="")
    segM = get_model(opt)
    segM.load_state_dict(checkpoint["model_state_dict"])
    segM = segM.to(device)
    segM.eval()

    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True)

    patch_shape = opt.patch_shape
    test_metrics = {"DSC":[], "HSD":[], "ICC":[], "ARI":[], "ASSD":[] }
    overall_seg_time = []
    overall_time = []
    predictions_path = join(opt.results_path,'predictions')
    with torch.no_grad():
        for p_idx, Dict in enumerate(dataloader):
            start_time = time.time()
            p_id, data, label = Dict["p_id"][0], Dict["data"], Dict["label"]
            data_header = dataset.get_header(p_idx)
            print("Found data of shape ", data.size())

            # Create results folder.
            res_path = join(predictions_path, p_id)
            if not os.path.exists(res_path):
                os.makedirs(res_path)

            start_seg = time.time()
            result = torch.zeros_like(data, dtype=float).squeeze(0).squeeze(0)
            counter = torch.zeros_like(result)
            iters = 0
            divisor = opt.divisor
            max_iters = 0

            # Simulate segmentation.
            for x in range(0, data.shape[-3] - (patch_shape[0] - patch_shape[0] // divisor),
                           patch_shape[0] // divisor):
                for y in range(0, data.shape[-2] - (patch_shape[1] - patch_shape[1] // divisor),
                               patch_shape[1] // divisor):
                    for z in range(0, data.shape[-1] - (patch_shape[2] - patch_shape[2] // divisor),
                                   patch_shape[2] // divisor):
                        max_iters += 1

            # Computing segmentation.
            print("\nStarted computing segmentation for ", p_id)
            for x in range(0, data.shape[-3] - (patch_shape[0] - patch_shape[0] // divisor),
                           patch_shape[0] // divisor):
                for y in range(0, data.shape[-2] - (patch_shape[1] - patch_shape[1] // divisor),
                               patch_shape[1] // divisor):
                    for z in range(0, data.shape[-1] - (patch_shape[2] - patch_shape[2] // divisor),
                                   patch_shape[2] // divisor):

                        x = min(x, data.shape[-3] - patch_shape[0])
                        y = min(y, data.shape[-2] - patch_shape[1])
                        z = min(z, data.shape[-1] - patch_shape[2])

                        # Crop data.
                        crop = data[..., x:x + patch_shape[0], y:y + patch_shape[1], z:z + patch_shape[2]]
                        
                        # Normalization.
                        crop = utils.normalize(crop, opt)

                        # Prediction.
                        outM = (segM(crop.unsqueeze(0).to(device)).squeeze(0).squeeze(0)).cpu()
                        outM = torch.sigmoid(outM)
                        outM = torch.where(outM < 0.5, torch.tensor(0), torch.tensor(1))
                        result[x:x + patch_shape[0], y:y + patch_shape[1], z:z + patch_shape[2]] += outM
                        counter[x:x + patch_shape[0], y:y + patch_shape[1], z:z + patch_shape[2]] += 1

                        iters += 1
                        print("\rComputing segmentation {:6.2f}% done".format((iters * 1) / max_iters * 100), end="", flush=True)
            seg_time = time.time() - start_seg
            overall_seg_time.append(seg_time)
            print(" in {}s".format(round(seg_time, 2)))

            print("Postprocessing... ", end="")
            # Take a threshold to make it binary.
            result_no_threshold = result.div(counter)
            result = torch.where(result_no_threshold >= torch.tensor(0.5), torch.tensor(1), torch.tensor(0))

            # Find connected components and only keep largest.
            try:
              result = keep_only_largest_connected_component(result)

              # Find centroid of segmentation.
              centroid = get_centroid(result)

              # Select patch with centroid as centroid of patch.
              x = min(max(0, centroid[0] - patch_shape[0]//2), data.shape[-3] - patch_shape[0])
              y = min(max(0, centroid[1] - patch_shape[1]//2), data.shape[-2] - patch_shape[1])
              z = min(max(0, centroid[2] - patch_shape[2]//2), data.shape[-1] - patch_shape[2])

              # Create crop with zeros in case that dimension of input image is smaller than path shape.
              crop = data[..., x:x+patch_shape[0], y:y+patch_shape[1], z:z+patch_shape[2]].unsqueeze(0).to(device)
              crop = utils.normalize(crop, opt)

              # Feed only important patch into generator.
              outM = segM(crop).squeeze(0).squeeze(0).cpu().detach()
              outM = torch.sigmoid(outM).numpy()
              outM_thres = np.where(outM < 0.5, 0, 1)

              result = torch.zeros_like(data.squeeze(0), dtype=torch.int8).numpy()
              result[x:x+patch_shape[0], y:y+patch_shape[1], z:z+patch_shape[2]] = outM_thres

            except:
              # If nothing was contoured during sliding window.
              result = torch.zeros_like(data.squeeze(0), dtype=torch.int8).numpy()


            print("done.", end="\n")

            # Store result.
            print("Writing to permanent memory... ", end="")
            start_write = time.time()
            if opt.prob_map:
              nrrd.write(res_path + "/medical-gan-prediction-prob-map.nrrd",
                  outM,
                  data_header,
              )
            nrrd.write(res_path + "/medical-gan-prediction.nrrd",
                result,
                data_header,
            )
            print("done in {}s.".format(round(time.time() - start_write, 2)))
            total_time = time.time() - start_time
            overall_time.append(total_time)

            # Compute validation metrics.
            print("Computing metrics...")
            print(result.shape, label.shape)
            test_metrics = utils.compute_metrices(input=torch.ByteTensor(np.expand_dims(result, axis=(0,1))), target=label.unsqueeze(0), metrices=test_metrics, opt=opt)
          
            # Write losses into file.
            print("DSC: {}, HSD: {}, ICC: {}, ARI: {}, ASSD: {}".
                  format(test_metrics["DSC"][-1], test_metrics["HSD"][-1], test_metrics["ICC"][-1], test_metrics["ARI"][-1], test_metrics["ASSD"][-1]
                  ))
            # Write final results into file.
            with open(join(opt.results_path, "results_detailed.csv"), 'a', newline='') as csvfile:
              writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
              writer.writerow([p_id, test_metrics["DSC"][-1], test_metrics["HSD"][-1], test_metrics["ICC"][-1], test_metrics["ARI"][-1], test_metrics["ASSD"][-1], seg_time, total_time])
        
        # Write final results into file.
        with open(join(opt.results_path, "results.csv"), 'a', newline='') as csvfile:
          writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
          writer.writerow(["Mean",
                        np.mean(test_metrics["DSC"]),
                        np.mean(test_metrics["HSD"]),
                        np.mean(test_metrics["ICC"]),
                        np.mean(test_metrics["ARI"]),
                        np.mean(test_metrics["ASSD"]),
                        np.mean(overall_seg_time),
                        np.mean(overall_time)])
          writer.writerow(["Median",
                        np.median(test_metrics["DSC"]),
                        np.median(test_metrics["HSD"]),
                        np.median(test_metrics["ICC"]),
                        np.median(test_metrics["ARI"]),
                        np.median(test_metrics["ASSD"]),
                        np.median(overall_seg_time),
                        np.median(overall_time)])
          writer.writerow(["Min",
                        np.min(test_metrics["DSC"]),
                        np.min(test_metrics["HSD"]),
                        np.min(test_metrics["ICC"]),
                        np.min(test_metrics["ARI"]),
                        np.min(test_metrics["ASSD"]),
                        np.min(overall_seg_time),
                        np.min(overall_time)])
          writer.writerow(["Max",
                        np.max(test_metrics["DSC"]),
                        np.max(test_metrics["HSD"]),
                        np.max(test_metrics["ICC"]),
                        np.max(test_metrics["ARI"]),
                        np.max(test_metrics["ASSD"]),
                        np.max(overall_seg_time),
                        np.max(overall_time)])
          writer.writerow(["Standard deviation",
                        np.std(test_metrics["DSC"]),
                        np.std(test_metrics["HSD"]),
                        np.std(test_metrics["ICC"]),
                        np.std(test_metrics["ARI"]),
                        np.std(test_metrics["ASSD"]),
                        np.std(overall_seg_time),
                        np.std(overall_time)])

if __name__ == "__main__":
    test()

