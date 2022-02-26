from os.path import join, isfile
import numpy as np
import torch, argparse, os
import SimpleITK as sitk
import torchio as tio
from natsort import natsorted
from os import listdir

class NoSplitDataset(torch.utils.data.Dataset):
  """
  Load all data but do not split it into train, validation and test data.
  """
  def __init__(self, opt):
      self.opt = opt
      self.data_root = opt.data_root
      self.patient_ids, self.data_paths, self.label_paths = self.read_data_paths()

      # Split data into training and validation set.
      self.data_idx = np.arange(len(self.patient_ids))

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
    return len(self.patient_ids)

  def __getitem__(self, idx):
    # Read data from memory.
    data = sitk.ReadImage(self.data_paths[idx])

    # Read label from memory.
    label = sitk.ReadImage(self.label_paths[idx])

    # Copy label to new path.
    newpath = join(self.opt.results_path, self.patient_ids[idx])
    if not os.path.exists(newpath):
      os.makedirs(newpath)
    sitk.WriteImage(sitk.Cast(label, sitk.sitkUInt8), join(newpath, "label.nrrd"))

    newpath = join(newpath, "data.nrrd")
    return data, newpath


def create_noisy_dataset(opt):
  # Check options.
  if opt.results_path == opt.data_root:
    raise ValueError("Data root and results path are the same! You would overwrite your existing dataset, so I stop here!")

  # Create noise pipeline.
  if opt.noise == "random":
    noise = tio.transforms.RandomNoise(std=opt.random_std)
  elif opt.noise == "motion":
    noise = tio.transforms.RandomMotion(num_transforms=opt.motion_transforms)
  elif opt.noise == "blur":
    noise = tio.transforms.RandomBlur(std=opt.blur_std)

  # Do not shuffle data.
  opt.no_shuffle = True

  dataset = NoSplitDataset(opt=opt)

  # Iterate over dataset.
  indices = np.arange(len(dataset))
  for idx in indices:
    data, results_path = dataset.__getitem__(idx=idx)

    data_t = sitk.GetArrayFromImage(data).transpose(2, 1, 0).astype(np.float32) # Transpose, because sitk uses different coordinate system than pynrrd.
    data_t = torch.FloatTensor(data_t)

    # Add noise.
    data_t = noise(data_t.unsqueeze(0)).squeeze(0)


    # Copy new image data to old sitk image.
    result_image = sitk.GetImageFromArray(data_t.numpy().transpose(2, 1, 0))
    result_image.CopyInformation(data)

    # Write new image to folder
    fileWriter = sitk.ImageFileWriter()
    fileWriter.SetUseCompression(True)
    fileWriter.SetFileName(results_path)
    fileWriter.Execute(result_image)

    print("{:.2f}% done.".format((idx + 1) / len(dataset) * 100), end="\r", flush=True)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--results_path", type=str, default="noisy_data", required=False, help="Path to store the results.")
  parser.add_argument("--data_root", type=str, default="train_and_test", required=True, help="Path to data.")
  parser.add_argument("--noise", type=str, default="random", choices=["random", "motion", "blur"], required=True, help="Select noise to add.")
  parser.add_argument("--blur_std", type=float, default=2, required=False, help="The amount of standard deviation to be applied to create blur noise.")
  parser.add_argument("--random_std", type=float, default=30, required=False, help="The amount of standard deviation to be applied to create random noise.")
  parser.add_argument("--motion_transforms", type=int, default=5, required=False, help="The number of transforms to be applied to create motion noise.")
  opt = parser.parse_args()

  print("Started to create new noisy dataset...")
  create_noisy_dataset(opt)
  print("100\% done.")