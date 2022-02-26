import os, torch, sys, utils, nrrd, random
import numpy as np
from os import listdir
from os.path import isfile, join
from natsort import natsorted
import SimpleITK as sitk

class Dataset(torch.utils.data.Dataset):
  """
  Loads data and corresponding label and returns pytorch float tensor.
  """
  def __init__(self, opt, training):
    self.opt = opt
    self.data_root = opt.data_root
    self.training = training
    self.patient_ids, self.data_paths, self.label_paths, self.mask_paths = self.read_data_paths()
    self.voxel_spacing = None

    # Split data into training and validation set.
    self.data_idx = np.arange(len(self.patient_ids))
    np.random.seed(2021)
    if not self.opt.no_shuffle == True:
      self.data_idx = np.random.permutation(self.data_idx)

    self.train_val_size = int(0.8*len(self.patient_ids))
    self.test_size = len(self.patient_ids) - self.train_val_size

    if opt.dataset_contains == "train_and_test":
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

    elif opt.dataset_contains == "train":
      self.train_size = int(0.8*len(self.patient_ids))
      self.val_size = len(self.patient_ids) - self.train_size
      self.train_idx = self.data_idx[:self.train_size]
      self.val_idx = self.data_idx[self.train_size:]

    # This variable is set, so that the main memory does not explode.
    if "store_loaded_data" in self.opt:
      if not self.opt.store_loaded_data:
          self.max_nr_of_files_to_store = 0
      else:
          self.max_nr_of_files_to_store = 170

    self.loadedFiles = {}

  def __len__(self):
    if self.training:
      return len(self.train_idx)
    else:
      return len(self.val_idx)

  def nr_of_patients(self):
    return self.__len__()

  def read_data_paths(self):
    """
    Reads data paths.
    """
    listdir()
    data_root = self.opt.data_root

    # Read patient ids.
    patient_ids = [f for f in listdir(
        data_root) if not isfile(join(data_root, f))]
    patient_ids = natsorted(patient_ids)

    # Read data and label paths.
    data_paths = []
    label_paths = []
    mask_paths = []

    for p_id in patient_ids:
      data_paths.append(join(data_root, p_id, "data.nrrd"))
      label_paths.append(join(data_root, p_id, "label.nrrd"))
      m_path = join(data_root, p_id, "mask.nrrd")
      if os.path.exists(m_path):
        mask_paths.append(m_path)
      else:
        mask_paths.append(None)

    return patient_ids, data_paths, label_paths, mask_paths

  def shuffle_patch_choice(self):
    """
    It is randomly decided for which patients only background patches
    shall bereturned.
    """
    # Randomly choose 20% of val patches to include only background.
    self.no_prostate_patch_idx = random.sample(list(self.val_idx), int(len(self.val_idx)*0.2))

  def different_spacing(self, spacing_1, spacing_2, tolerance=0.0001):
    """
    Checks whether the spacings match with a tolerance.
    """
    if abs(spacing_1[0]-spacing_2[0]) > tolerance:
        return True
    if abs(spacing_1[1]-spacing_2[1]) > tolerance:
        return True
    if abs(spacing_1[2]-spacing_2[2]) > tolerance:
        return True
    return False


  def __getitem__(self, idx):
    """
    Read data and label and return them.
    """
    idx = self.train_idx[idx] if self.training else self.val_idx[idx]
    p_id = self.patient_ids[idx]

    # return already loaded data
    if p_id not in self.loadedFiles:
    
      # Read data from memory.
      data = sitk.ReadImage(self.data_paths[idx])

      # Read label from memory.
      label = sitk.ReadImage(self.label_paths[idx])

      # Get the voxel spacing of the data and the label.
      data_spacing = data.GetSpacing()
      label_spacing = label.GetSpacing()

      # Check whether the input and ground truth have same spacing.
      if self.different_spacing(data_spacing, label_spacing):
        print("The spacing of the input is: {}. ".format(data_spacing))
        print("The spacing of the label is: {}. ".format(label_spacing))
        raise Exception("The spacing of data and label don't match.")

      # Set the voxel spacing of the dataset.
      if self.voxel_spacing is None:
        self.voxel_spacing = data_spacing

      # Make sure that all samples of the dataset have the same spacing.
      elif self.different_spacing(self.voxel_spacing, data_spacing):
        print("The spacing of the previous input is: {}. ".format(self.voxel_spacing))
        print("The spacing of the current input is: {}. ".format(data_spacing))
        raise Exception("The spacing of current input and previous input don't match.")

      data = sitk.GetArrayFromImage(data).transpose(2, 1, 0).astype(np.float32) # Transpose, because sitk uses different coordinate system than pynrrd.
      label = sitk.GetArrayFromImage(label).transpose(2, 1, 0).astype(np.uint8) # Transpose, because sitk uses different coordinate system than pynrrd.
      label[label > 0] = 1 # Ensure that all the labels are between 0 and 1.
      
      # Clip data to 0.5 and 99.5 percentiles.
      if not self.opt.no_clip == True:
        low, high = np.percentile(data, [0.5, 99.5])
        data = np.clip(data, low, high)

      # COnvert numpy to torch format.
      data = torch.FloatTensor(data)
      label = torch.ByteTensor(label)

      # Real mask if exists and cut label.
      if self.mask_paths[idx] is not None:
        mask = sitk.ReadImage(self.mask_paths[idx])
        mask = sitk.GetArrayFromImage(mask).transpose(2, 1, 0).astype(np.uint8) # Transpose, because sitk uses different coordinate system than pynrrd.
        mask = torch.ByteTensor(mask)
        label = torch.where(mask == 1, label, torch.tensor(0.))

      # Store already loaded files in RAM.
      if self.max_nr_of_files_to_store > 0:
          self.loadedFiles[p_id] = [p_id, data, label]
          self.max_nr_of_files_to_store -= 1

    else:
      p_id, data, label = self.loadedFiles[p_id]

    if self.training:
      data, label = utils.select_random_patches(data, label, self.opt)
    else:
      # empty = True if idx in self.no_prostate_patch_idx else False
      empty = False # Out-comment this and un-comment previous line to sample 20% of validation patches empty.
      data, label = utils.select_random_patch(data, label, self.opt, empty)

    if self.opt.switch == True:
      # Normalization.
      data = utils.normalize(data, self.opt)

    # Data augmentation.
    if not self.opt.no_augmentation == True and self.training:
      data, label = utils.data_augmentation_batch(data, label, self.opt)

    if self.opt.switch == False:
      # Normalization.
      data = utils.normalize(data, self.opt)

    #assert (torch.max(label) <= 1 and torch.min(label) >= 0)

    return p_id, data, label, self.data_idx[idx]
