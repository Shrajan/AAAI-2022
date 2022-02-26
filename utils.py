import numpy as np
import matplotlib.pyplot as plt
import os, torch, medpy, sys, torch, random
from medpy.metric.binary import hd, dc, asd, assd, precision, sensitivity, specificity
import SimpleITK as sitk
from collections import OrderedDict

import elasticdeform
import torchio as tio
from sklearn.metrics.cluster import pair_confusion_matrix

####################################################
#----------------- OPTIONAL STUFF -----------------#
####################################################

def str_to_bool(value):
    """
    Turns a string into boolean value.
    """
    t = ['true', 't', '1', 'y', 'yes', 'ja', 'j']
    f = ['false', 'f', '0', 'n', 'no', 'nein']
    if value.lower() in t:
        return True
    elif value.lower() in f:
        return False
    else:
        raise ValueError("{} is not a valid boolean value. Please use one out of {}".format(value, t + f))

def overwrite_request(path):
    if os.path.exists(path):
        valid = False
        while not valid:
            answer = input("{} already exists. Are you sure you want to overwrite everything in this folder? [yes/no]\n".format(path))
            if str_to_bool(answer):
                valid = True
            elif not str_to_bool(answer):
                sys.exit(1)

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("Created folder(s) {}".format(path))
    else:
        print("Folder(s) {} already exist(s).".format(path))

def plot_losses(opt, path, title, xlabel, ylabel, plot_name, *args, axis="auto"):
    """
    Creates nice plots and saves them as PNG files onto permanent memory.
    """
    fig = plt.figure()
    plt.title(title)
    for element in args:
        plt.plot(element[0], label=element[1], alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis(axis)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(path, plot_name))
    plt.close(fig)


####################################################
#------------------ DATA LOADING ------------------#
####################################################

def normalize(data, opt):
  """
  Normalize data with method that is defined in opt.
  """
  if opt.normalize == "local":
    data = (data - data.mean() + 1e-8) / (data.std() + 1e-8) # z-score
  elif opt.normalize == "global":
    std = 403.24596999234836
    mean = -774.3474039638812
    data = data - mean
    data = data / std
  elif opt.normalize == "-11":
    data -= data.min()
    data /= data.max()
    data = (data - 0.5) / 0.5 # -> [-1, 1]
  elif opt.normalize.lower() != "none":
    sys.exit("Normalize parameter must be one out of {None, local, global, -11}")
  return data

def select_random_patches(data, label, opt):
  """
  Generate random patches from 3D tensors
  """
  patch_shape = [opt.patch_shape[0], opt.patch_shape[1], opt.patch_shape[2]]
  if not opt.no_augmentation:
    patch_shape[0] += opt.margin
    patch_shape[1] += opt.margin
    patch_shape[2] += opt.margin

  n_patches = 1
  x, y, z = data.shape[-3], data.shape[-2], data.shape[-1]

  data_batch = torch.zeros((n_patches, patch_shape[0], patch_shape[1], patch_shape[2]))
  label_batch = torch.zeros((n_patches, patch_shape[0], patch_shape[1], patch_shape[2]))

  if opt.img_mod == "CT":
    # For CT images either the complete foreground volume is included or excluded.
    nonzeros = torch.nonzero(label)
    maxX, maxY, maxZ = nonzeros[:, 0].max(), nonzeros[:, 1].max(), nonzeros[:, 2].max()
    minX, minY, minZ = nonzeros[:, 0].min(), nonzeros[:, 1].min(), nonzeros[:, 2].min()

    for idx in range(n_patches):
        r = torch.rand(1)
        if r < opt.p_foreground:
          # Generate lower borders for random patch generation
          l_x = torch.randint(low=min(minX - (patch_shape[0]-(maxX-minX)), x-patch_shape[0]), high=min(minX, x-patch_shape[0]), size=(1,), dtype=torch.int16)[0]
          l_y = torch.randint(low=min(minY - (patch_shape[1]-(maxY-minY)), y-patch_shape[1]), high=min(minY,y-patch_shape[1]), size=(1,), dtype=torch.int16)[0]
          l_z = torch.randint(low=min(minZ - (patch_shape[2]-(maxZ-minZ)), z-patch_shape[2]), high=min(minZ, z-patch_shape[2]), size=(1,), dtype=torch.int16)[0]
          
          l_x = max(l_x, 0)
          l_y = max(l_y, 0)
          l_z = max(l_z, 0)

          l_batch = label[l_x:l_x + patch_shape[0], l_y:l_y + patch_shape[1], l_z:l_z + patch_shape[2]]
        else:
          is_empty = False
          counter = 0
          while not is_empty:
            l_x = torch.randint(low=0, high=x-patch_shape[0], size=(1,), dtype=torch.int16)[0]
            l_y = torch.randint(low=0, high=y-patch_shape[1], size=(1,), dtype=torch.int16)[0]
            l_z = torch.randint(low=0, high=z-patch_shape[2], size=(1,), dtype=torch.int16)[0]

            l_batch = label[l_x:l_x + patch_shape[0], l_y:l_y + patch_shape[1], l_z:l_z + patch_shape[2]]

            if not l_batch.bool().any():
              is_empty = True

            counter += 1
          
        d_batch = data[l_x:l_x + patch_shape[0], l_y:l_y + patch_shape[1], l_z:l_z + patch_shape[2]]
        label_batch[idx, ...] = l_batch
        data_batch[idx, ...] = d_batch
  else:
    # For MR modality patches are sampled completely randomly.

    for idx in range(n_patches):
      r = torch.rand(1)
      if r < opt.p_foreground:
        c_x, c_y, c_z = x/2, y/2, z/2
        l_x = int(c_x - patch_shape[0] / 2)
        l_y = int(c_y - patch_shape[1] / 2)
        l_z = int(c_z - patch_shape[2] / 2)
        l_batch = label[l_x:l_x + patch_shape[0], l_y:l_y + patch_shape[1], l_z:l_z + patch_shape[2]]
        d_batch = data[l_x:l_x + patch_shape[0], l_y:l_y + patch_shape[1], l_z:l_z + patch_shape[2]]
        label_batch[idx, ...] = l_batch
        data_batch[idx, ...] = d_batch
      
      else:
        l_x = torch.randint(low=0, high=x-patch_shape[0], size=(1,), dtype=torch.int16)[0]
        l_y = torch.randint(low=0, high=y-patch_shape[1], size=(1,), dtype=torch.int16)[0]
        l_z = torch.randint(low=0, high=z-patch_shape[2], size=(1,), dtype=torch.int16)[0]

        l_batch = label[l_x:l_x + patch_shape[0], l_y:l_y + patch_shape[1], l_z:l_z + patch_shape[2]]
        d_batch = data[l_x:l_x + patch_shape[0], l_y:l_y + patch_shape[1], l_z:l_z + patch_shape[2]]
        label_batch[idx, ...] = l_batch
        data_batch[idx, ...] = d_batch

  return data_batch, label_batch

def select_random_patch(data, label, opt, empty):
  """
  Either return a patch with prostate included or not.
  """
  patch_shape = opt.patch_shape
  n_patches = 1
  x, y, z = data.shape[-3], data.shape[-2], data.shape[-1]

  data_batch = torch.zeros((n_patches, patch_shape[0], patch_shape[1], patch_shape[2]))
  label_batch = torch.zeros((n_patches, patch_shape[0], patch_shape[1], patch_shape[2]))

  if opt.img_mod == "CT":
    nonzeros = torch.nonzero(label)
    maxX, maxY, maxZ = nonzeros[:, 0].max(), nonzeros[:, 1].max(), nonzeros[:, 2].max()
    minX, minY, minZ = nonzeros[:, 0].min(), nonzeros[:, 1].min(), nonzeros[:, 2].min()

    # Shall prostate be included in this patch?
    for idx in range(n_patches):
      if not empty==True:
        # Generate lower borders for random patch generation
        l_x = torch.randint(low=min(minX - (patch_shape[0]-(maxX-minX)), x-patch_shape[0]), high=min(minX, x-patch_shape[0]), size=(1,), dtype=torch.int16)[0]
        l_y = torch.randint(low=min(minY - (patch_shape[1]-(maxY-minY)), y-patch_shape[1]), high=min(minY,y-patch_shape[1]), size=(1,), dtype=torch.int16)[0]
        l_z = torch.randint(low=min(minZ - (patch_shape[2]-(maxZ-minZ)), z-patch_shape[2]), high=min(minZ, z-patch_shape[2]), size=(1,), dtype=torch.int16)[0]
        
        l_x = max(l_x, 0)
        l_y = max(l_y, 0)
        l_z = max(l_z, 0)

        l_batch = label[l_x:l_x + patch_shape[0], l_y:l_y + patch_shape[1], l_z:l_z + patch_shape[2]]
      else:
        is_empty = False
        counter = 0
        while not is_empty:
          l_x = torch.randint(low=0, high=x-patch_shape[0], size=(1,), dtype=torch.int16)[0]
          l_y = torch.randint(low=0, high=y-patch_shape[1], size=(1,), dtype=torch.int16)[0]
          l_z = torch.randint(low=0, high=z-patch_shape[2], size=(1,), dtype=torch.int16)[0]

          l_batch = label[l_x:l_x + patch_shape[0], l_y:l_y + patch_shape[1], l_z:l_z + patch_shape[2]]

          if not l_batch.bool().any():
            is_empty = True

          counter += 1

      d_batch = data[l_x:l_x + patch_shape[0], l_y:l_y + patch_shape[1], l_z:l_z + patch_shape[2]]
      label_batch[idx, ...] = l_batch
      data_batch[idx, ...] = d_batch
  else:
    for idx in range(n_patches):
      c_x, c_y, c_z = x/2, y/2, z/2
      l_x = int(c_x - patch_shape[0] / 2)
      l_y = int(c_y - patch_shape[1] / 2)
      l_z = int(c_z - patch_shape[2] / 2)
      l_batch = label[l_x:l_x + patch_shape[0], l_y:l_y + patch_shape[1], l_z:l_z + patch_shape[2]]
      d_batch = data[l_x:l_x + patch_shape[0], l_y:l_y + patch_shape[1], l_z:l_z + patch_shape[2]]
      label_batch[idx, ...] = l_batch
      data_batch[idx, ...] = d_batch

  return data_batch, label_batch


def data_augmentation_batch(data_batch, label_batch, opt):
  """
  With a probability of 50% perform elasic deformation
  and with probability of 50% perform flip over x-axis.
  """
  # Define input subject.
  input_subject = tio.Subject({'data': tio.ScalarImage(tensor=data_batch), 'label': tio.LabelMap(tensor=label_batch)})

  # Define flip transform.
  flip_transform_LR = tio.RandomFlip(axes=('LR',), flip_probability=0.5) # Lateral flip
  flip_transform_AP = tio.RandomFlip(axes=('AP',), flip_probability=0.5) # Anterior posterior flip

  # Crop patch, because opt.margin voxels more were sampled per axis in the sampling steps.
  crop = tio.Crop(int(opt.margin/2))

  # Define spatial transformations.
  spatial_transforms = {
    tio.RandomAffine(
      scales=(0.8, 1.2),
      degrees=15,
      translation=(-5,5),
      isotropic=False,
      center='image',
      default_pad_value='mean',
      image_interpolation='linear'
    ): 0.25,
    tio.RandomAffine(
      scales=(0.8, 1.2),
      degrees=15,
      translation=(-5,5),
      isotropic=False,
      center='image',
      default_pad_value='mean',
      image_interpolation='bspline'
    ): 0.25,
    tio.RandomElasticDeformation(
      num_control_points=7,
      max_displacement=5,
      locked_borders=2,
      image_interpolation='linear'
    ): 0.25,
    tio.RandomElasticDeformation(
      num_control_points=7,
      max_displacement=7.5,
      locked_borders=2,
      image_interpolation='bspline'
    ): 0.25
  }

  # Compose transforms.
  transforms = tio.Compose([
    flip_transform_LR,
    flip_transform_AP,
    tio.OneOf(spatial_transforms, p=0.8),
    crop
  ])

  # Perform transformations.
  output_subject = transforms(input_subject)

  data_batch = output_subject['data'].data
  label_batch = output_subject['label'].data

  return data_batch, label_batch


####################################################
#-------------- TRAINING PARAMETERS ---------------#
####################################################

def set_optimizer(opt, model_params):
    if opt.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model_params, lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps, weight_decay=opt.weight_decay)
    elif opt.optimizer.lower() == "adamax":
        optimizer = torch.optim.Adamax(model_params, lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps, weight_decay=opt.weight_decay)
    else:
        sys.exit("{} is not a valid optimizer. Choose one of {adam, adamax}".format(opt.optimizer))
    return optimizer

class BinaryDiceLoss(torch.nn.Module):
  """
  Generalized Dice Loss for binary case (Only 0 and 1 in ground truth labels.)

  https://arxiv.org/abs/1707.03237
  """
  def __init__(self, smooth=0.000001):
    super(BinaryDiceLoss, self).__init__()
    self.smooth = smooth

  def forward(self, y_pred, y_true):
    y_pred = torch.sigmoid(y_pred)   # The output from the model is logits.

    # Original Dice Loss.
    dividend = torch.mul(y_pred, y_true).sum() + self.smooth
    divisor = (y_pred + y_true).sum() + self.smooth
    loss1 = dividend / divisor

    # Inverse Dice Loss.
    inv_y_pred = 1 - y_pred
    inv_y_true = 1 - y_true

    dividend = torch.mul(inv_y_pred, inv_y_true).sum() + self.smooth
    divisor = (inv_y_pred + inv_y_true).sum() + self.smooth
    loss2 = dividend / divisor

    return 1 - loss1 - loss2 # Because it's a loss...

class BCE_Dice_Loss(torch.nn.Module):
  def __init__(self):
    super(BCE_Dice_Loss, self).__init__()
    self.bce_loss_func = torch.nn.BCEWithLogitsLoss()
    self.dice_loss_func = BinaryDiceLoss()

  def forward(self, y_pred, y_true):
    bce_loss  = self.bce_loss_func(y_pred, y_true)
    dice_loss = self.dice_loss_func(y_pred, y_true)
    return bce_loss + dice_loss

def set_loss_fn(opt):
    if opt.loss_fn == "binary_cross_entropy":
        return torch.nn.BCEWithLogitsLoss()
    elif opt.loss_fn == "dice":
        return BinaryDiceLoss()
    elif opt.loss_fn.lower() == "bce_dice_loss":
        return BCE_Dice_Loss()
    return loss_fn

####################################################
#-------------- VALIDATION METRICES ---------------#
####################################################

def adjusted_rand_index(labels_true, labels_pred):
    (tn, fp), (fn, tp) = pair_confusion_matrix(labels_true, labels_pred)

    # Special cases: empty data or full agreement
    if fn == 0 and fp == 0:
        return 1.0

    tn = tn.astype(np.float64)
    tp = tp.astype(np.float64)
    fp = fp.astype(np.float64)
    fn = fn.astype(np.float64)

    return 2. * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))

# Adjusted rand index/scrore.
def ari(result, reference):
    labels_true = reference.flatten()
    labels_pred = result.flatten()
    return adjusted_rand_index(labels_true, labels_pred)      # Adjusted rand index/scrore.


def interclass_correlation(label_true, label_pred):
    mean_label_true = np.mean(label_true)
    mean_label_pred = np.mean(label_pred)
    grandmean = (mean_label_true + mean_label_pred)/2
    numberElements = len(label_pred)

    m = np.mean([label_true, label_pred], axis=0)
    ssw = np.sum(np.sum([np.power(label_true - m, 2), np.power(label_pred - m, 2)], axis=0), axis=0)
    ssb = np.sum(np.power(m - grandmean, 2), axis=0)

    old_settings = np.seterr(divide='print', invalid='ignore')

    ssw = ssw / numberElements
    ssb = ssb / (numberElements - 1) * 2
    num = ssb - ssw
    den = ssb + ssw
    icc = num / (den + 0.0000001)
    return icc

# Interclass correlation.
def icc(result, reference):
    labels_true = reference.flatten()
    labels_pred = result.flatten()
    return interclass_correlation(labels_true, labels_pred)      # Interclass correlation.

# Compute validation losses.
def compute_metrices(input, target, metrices, opt=None):

    n_batches = target.shape[1]
    
    for b in range(n_batches):
      i = input[b].squeeze(0).cpu().detach().numpy()
      t = target[b].squeeze(0).cpu().detach().numpy()

      i = np.where(i < 0.5, 0, 1)
      t = np.where(t < 0.5, 0, 1)

      nz_input = np.count_nonzero(i)
      nz_target = np.count_nonzero(t)

      if 0 == nz_input and 0 == nz_target:
          #m1 = 1.0
          m2 = 0.0
          m3 = 0.0
          m4 = 0.0
          m5 = 1.0

      elif 0 != nz_input and 0 == nz_target:
          #m1 = 0.0
          m2 = 50.0
          m3 = 25.0
          m4 = 25.0
          m5 = 0.0

      elif 0 == nz_input and 0 != nz_target:
          #m1 = 0.0
          m2 = 50.0
          m3 = 25.0
          m4 = 25.0
          m5 = 0.0

      else:
          if opt == None:
            m2 = hd(result=i, reference=t)                                          # Hausdorff Distance.
            m5 = assd(result=i, reference=t)                                        # Average symmetric surface distance.
          else:
            m2 = hd(result=i, reference=t, voxelspacing = opt.voxel_spacing)        # Hausdorff Distance with voxel spacing.
            m5 = assd(result=i, reference=t, voxelspacing = opt.voxel_spacing)      # Average symmetric surface distance with voxel spacing.           
          m3 = icc(result=i, reference=t)                                           # Interclass correlation.
          m4 = ari(result=i, reference=t)                                           # Adjusted rand index/scrore.
  
      m1 = dc(result=i, reference=t)                                                 # Dice Coefficient.
      metrices["DSC"].append(m1)
      metrices["HSD"].append(m2)
      metrices["ICC"].append(m3)
      metrices["ARI"].append(m4)
      metrices["ASSD"].append(m5)

    return metrices

###############################################################
#---------------------- EXTRA FUNCTIONS ----------------------#
###############################################################

def random_flip(data, lesion):
    if torch.rand(1) < 0.5:
        data = data.flip(dims=(1,))
        lesion = lesion.flip(dims=(1,))
    return data, lesion

def random_scaling(opt, data, lesion):
    old_shape = data[0].shape
    final_shape = 64
    interpolation_mode = opt.interpolation_mode

    # Computing random scale factor.
    max_scale_factor = data.shape[1] / final_shape
    
    scale_factor_x = np.random.uniform(1 - ((1-max_scale_factor)/2), 1 + ((1-max_scale_factor)/2))
    scale_factor_y = np.random.uniform(1 - ((1-max_scale_factor)/2), 1 + ((1-max_scale_factor)/2))
    scale_factor_z = np.random.uniform(1 - ((1-max_scale_factor)/2), 1 + ((1-max_scale_factor)/2))

    # Scaling.
    pet = torch.nn.functional.interpolate(data[0].unsqueeze(0).unsqueeze(0),\
        scale_factor=(scale_factor_x, scale_factor_y, scale_factor_z), mode=interpolation_mode).squeeze().squeeze()
    prostate_contour = torch.nn.functional.interpolate(data[1].unsqueeze(0).unsqueeze(0),\
        scale_factor=(scale_factor_x, scale_factor_y, scale_factor_z), mode=interpolation_mode).squeeze().squeeze()
    lesion = torch.nn.functional.interpolate(lesion.unsqueeze(0).unsqueeze(0),\
        scale_factor=(scale_factor_x, scale_factor_y, scale_factor_z), mode=interpolation_mode).squeeze().squeeze()

    # Cropping into old shape.
    minx = 0
    maxx = pet.shape[0]
    miny = 0
    maxy = pet.shape[1]
    minz = 0
    maxz = pet.shape[2]

    if old_shape[0] > final_shape:
        if (maxx - minx) % 2 != 0:
            maxx -= 1
        if (maxy - miny) % 2 != 0:
            maxy -= 1
        if (maxz - minz) % 2 != 0:
            maxz -= 1
    else:
        if (maxx - minx) % 2 != 0:
            maxx += 1
        if (maxy - miny) % 2 != 0:
            maxy += 1
        if (maxz - minz) % 2 != 0:
            maxz += 1

    """
    Choose all tensors to have size of 64x64x64
    """
    while maxx - minx > final_shape:
        maxx -= 1
        minx += 1

    while maxy - miny > final_shape:
        maxy -= 1
        miny += 1
        
    while maxz - minz > final_shape:
        maxz -= 1
        minz += 1

    pet = pet[minx:maxx,miny:maxy,minz:maxz]
    prostate_contour = prostate_contour[minx:maxx,miny:maxy,minz:maxz]
    data = torch.cat((pet.unsqueeze(0), prostate_contour.unsqueeze(0)), dim=0)
    lesion = lesion[minx:maxx,miny:maxy,minz:maxz]

    # Correct the lesion to have only int values.
    lesion = np.where(lesion < 0.5, 0, lesion)
    lesion = np.where(np.logical_and(0.5 <= lesion, lesion < 1.5), 1, lesion)
    lesion = np.where(1.5 <= lesion, 2, lesion)
    lesion = torch.tensor(lesion)
    return data, lesion


def data_augmentation(data, label):
  # Random flip.
  if torch.rand(1) < 0.5:
    data = data.flip(dims=(0,))
    label = label.flip(dims=(0,))

  # Random elastic deformation.
  [data, label] = elasticdeform.deform_random_grid([data.numpy(), label.numpy()], mode='mirror', sigma=0.5, order=[3,0])
  data, label = torch.FloatTensor(data), torch.FloatTensor(label)
  return data, label

def label_batch_data_augmentation(label):
  # Random elastic deformation only for labels maps.
  transform = tio.Compose([
    tio.OneOf({
    tio.RandomElasticDeformation(num_control_points=7,max_displacement=5,locked_borders=2,image_interpolation='nearest'): 0.5,
    tio.RandomAffine(
    scales=(0.5, 1.8),
    degrees=45,
    translation=5,
    isotropic=False,
    center="image",
    default_pad_value="minimum",
    image_interpolation='nearest'): 0.5
  }, p=1.0)
  ])
  out = torch.zeros_like(label)
  for l_idx, l in enumerate(label):
    out[l_idx] = transform(l)
  return out.float()


  

def rescale(data, lesion):
    PET = data[0]
    final_shape = 64
    minx = 0
    maxx = PET.shape[0]
    miny = 0
    maxy = PET.shape[1]
    minz = 0
    maxz = PET.shape[2]

    if (maxx - minx) % 2 != 0:
        maxx -= 1
    if (maxy - miny) % 2 != 0:
        maxy -= 1
    if (maxz - minz) % 2 != 0:
        maxz -= 1

    """
    Choose all tensors to have size of 64x64x64
    """
    while maxx - minx > final_shape:
        maxx -= 1
        minx += 1

    while maxy - miny > final_shape:
        maxy -= 1
        miny += 1
        
    while maxz - minz > final_shape:
        maxz -= 1
        minz += 1

    PET = PET[minx:maxx,miny:maxy,minz:maxz]
    prostate_contour = data[1]
    prostate_contour = prostate_contour[minx:maxx,miny:maxy,minz:maxz]
    data = torch.cat((PET.unsqueeze(0), prostate_contour.unsqueeze(0)), dim=0)
    lesion = lesion[minx:maxx,miny:maxy,minz:maxz]

    return data, lesion

def fit_prediction_to_pet(pet_shape, pred, minx, maxx, miny, maxy, minz, maxz, opt):
    t = np.full(pet_shape, 0, dtype=float)

    t[minx:maxx, miny:maxy, minz:maxz] = pred

    return t

def rescale_array(data, new_shape):
  data = torch.nn.functional.interpolate(data.unsqueeze(0).unsqueeze(0),\
    size=new_shape, mode="trilinear", align_corners=True).squeeze().squeeze()
  return data

def resample_prostate(prostate_contour, new_shape):
  prostate_contour = torch.nn.functional.interpolate(prostate_contour.unsqueeze(0).unsqueeze(0),\
    size=new_shape, mode="nearest").squeeze().squeeze()
  return prostate_contour

def resample_data(data, header, interpolation="trilinear"):
    """
    @data will be resampled to the header.
    """
    # Resample data.
    resample = sitk.ResampleImageFilter()
    if interpolation == "trilinear":
        resample.SetInterpolator(sitk.sitkLinear)
    elif interpolation == "nearest":
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    resample.SetOutputDirection(header.GetDirection())
    resample.SetOutputOrigin(header.GetOrigin())

    new_spacing = header.GetSpacing()
    resample.SetOutputSpacing(new_spacing)

    orig_size = np.array(data.GetSize(), dtype=np.int)
    orig_spacing = data.GetSpacing()
    new_size = orig_size*([i / j for i, j in zip(list(orig_spacing),new_spacing)])
    new_size = np.ceil(new_size).astype(np.int)
    new_size = [int(s) for s in new_size]
    resample.SetSize(new_size)

    newimage = resample.Execute(data)
    data = sitk.GetArrayFromImage(newimage)
    data = data.transpose(2, 1, 0) # Transpose, because sitk uses different coordinate system than pynrrd.

    return data

def rescale_data(data, new_shape, header, interpolation="trilinear"):
  """
  Data will be rescaled to new shape
    @new_shape: new shape of tensor
    @interpolation: interpolation method = {"trilinear", "nearest"}
    @header: the header of the file
  """
  resample = sitk.ResampleImageFilter()
  if interpolation == "trilinear":
      resample.SetInterpolator(sitk.sitkLinear)
  elif interpolation == "nearest":
      resample.SetInterpolator(sitk.sitkNearestNeighbor)
  resample.SetOutputDirection(header.GetDirection())
  resample.SetOutputOrigin(header.GetOrigin())
  resample.SetOutputSpacing((1.0, 1.0, 1.0))
  resample.SetSize(new_shape)

  newimage = resample.Execute(data)
  data = sitk.GetArrayFromImage(newimage)
  data = data.transpose(2, 1, 0) # Transpose, because sitk uses different coordinate system than pynrrd.

  return data

def convert_sitk_to_pynrrd_header(header):
    new_h = OrderedDict()
    new_h["type"] = "double"
    new_h["dimension"] = 3
    new_h["space"] ="left-posterior-superior"
    new_h["sizes"] = np.array(header.GetSize())
    new_h["encoding"] = "gzip"
    new_h["space origin"] = np.array(header.GetOrigin())
    spacing = np.zeros((3, 3))
    s = header.GetSpacing()
    spacing[0, 0] = s[0]
    spacing[1, 1] = s[1]
    spacing[2, 2] = s[2]
    new_h["space directions"] = spacing
    new_h["kinds"] = ['domain', 'domain', 'domain']
    new_h["endian"] = "little"
    return new_h

def shear(data):
    """
    A function to shear data.
    """
    sps = 3 # Splits per side
    sl = (int(data.shape[-3]/sps),int(data.shape[-2]/sps),int(data.shape[-1]/sps)) # Side lengths.
    sheared_t = torch.zeros((int(sps**3), sl[0], sl[1], sl[2]))

    l = 0
    for i in range(sps):
        for j in range(sps):
            for k in range(sps):
                sheared_t[l, :, :, :]  = data[i*sl[0]:(i+1)*sl[0], j*sl[1]:(j+1)*sl[1], k*sl[2]:(k+1)*sl[2]]
                l += 1
    return sheared_t

def combine(data):
    sps = 3 # Splits per side
    sl = (int(data.shape[-3]),int(data.shape[-2]),int(data.shape[-1])) # Side lengths.
    sl_new = tuple([tmp * sps for tmp in sl])

    combined_t = torch.zeros(sl_new)

    l = 0
    for i in range(sps):
        for j in range(sps):
            for k in range(sps):
                combined_t[i*sl[0]:(i+1)*sl[0], j*sl[1]:(j+1)*sl[1], k*sl[2]:(k+1)*sl[2]] = data[l,:,:,:]
                l += 1

    return combined_t


def divide_data_into_patches(data, label, opt):
  """
  Divide data into patches that fit into the model.
  """
  patch_shape = opt.patch_shape
  n_patches = 1

  x, y, z = data.shape[-3]//patch_shape[0], data.shape[-2]//patch_shape[1], data.shape[-1]//patch_shape[2]

  data_batch = torch.zeros((x*y*x, patch_shape[0], patch_shape[1], patch_shape[2]))
  label_batch = torch.zeros_like(data_batch)

  idx = 0
  for i in range(x):
    for j in range(y):
      for k in range(z):
        data_batch[idx, ...] = data[i*patch_shape[0]:(i+1)*patch_shape[0], j*patch_shape[1]:(j+1)*patch_shape[1], k*patch_shape[2]:(k+1)*patch_shape[2]]
        idx += 1

  for idx, (i, j, k) in enumerate(zip(range(x), range(y), range(z))):
    label_batch[idx, ...] = label[i*patch_shape[0]:(i+1)*patch_shape[0], j*patch_shape[1]:(j+1)*patch_shape[1], k*patch_shape[2]:(k+1)*patch_shape[2]]

  return data_batch, label_batch

def combine_data_from_patches(label_batch, old_shape, opt):
  patch_shape = opt.patch_shape
  n_patches = 1

  label = torch.zeros(old_shape)

  x, y, z = label.shape[-3]//patch_shape[0], label.shape[-2]//patch_shape[1], label.shape[-1]//patch_shape[2]

  idx = 0
  for i in range(x):
    for j in range(y):
      for k in range(z):
        label[i*patch_shape[0]:(i+1)*patch_shape[0], j*patch_shape[1]:(j+1)*patch_shape[1], k*patch_shape[2]:(k+1)*patch_shape[2]] = label_batch[idx, ...]
        idx += 1

  return label

def get_one_patch(data, label, opt):
  """
  Crop data and label tensor to region of interest and return one tensor of patch_shape shape.
  """
  patch_shape = opt.patch_shape
  data, label = data.squeeze(0).squeeze(0), label.squeeze(0).squeeze(0)

  x, y, z = data.shape[-3], data.shape[-2], data.shape[-1]

  nonzeros = torch.nonzero(label)
  maxX, maxY, maxZ = 0, 0, 0
  minX, minY, minZ = label.shape[0], label.shape[1], label.shape[2]
  for nzr in nonzeros:
    if (nzr[0] > maxX): maxX = nzr[0]
    if (nzr[0] < minX): minX = nzr[0]
    if (nzr[1] > maxY): maxY = nzr[1]
    if (nzr[1] < minY): minY = nzr[1]
    if (nzr[2] > maxZ): maxZ = nzr[2]
    if (nzr[2] < minZ): minZ = nzr[2]

  l_x = torch.randint(low=min(minX - (patch_shape[0]-(maxX-minX)), x-patch_shape[0]), high=min(minX, x-patch_shape[0]), size=(1,), dtype=torch.int16)[0]
  l_y = torch.randint(low=min(minY - (patch_shape[1]-(maxY-minY)), y-patch_shape[1]), high=min(minY,y-patch_shape[1]), size=(1,), dtype=torch.int16)[0]
  l_z = torch.randint(low=min(minZ - (patch_shape[2]-(maxZ-minZ)), z-patch_shape[2]), high=min(minZ, z-patch_shape[2]), size=(1,), dtype=torch.int16)[0]

  l_x = max(l_x, 0)
  l_y = max(l_y, 0)
  l_z = max(l_z, 0)

  l_batch = label[l_x:l_x + patch_shape[0], l_y:l_y + patch_shape[1], l_z:l_z + patch_shape[2]]
  d_batch = data[l_x:l_x + patch_shape[0], l_y:l_y + patch_shape[1], l_z:l_z + patch_shape[2]]

  return d_batch.unsqueeze(0).unsqueeze(0), l_batch.unsqueeze(0).unsqueeze(0)

def resample_to_new_voxel_spacing(data, header, new_spacing):
  """
  Takes @data with header @header and resamples it to new
  voxel spacing @new_spacing.
  """
  resample = sitk.ResampleImageFilter()
  resample.SetInterpolator(sitk.sitkLinear)
  resample.SetOutputDirection(header.GetDirection())
  resample.SetOutputOrigin(header.GetOrigin())
  resample.SetOutputSpacing(new_spacing)

  orig_size = np.array(data.GetSize(), dtype=np.int)
  orig_spacing = data.GetSpacing()
  new_size = orig_size*([i / j for i, j in zip(list(orig_spacing),new_spacing)])
  new_size = np.ceil(new_size).astype(np.int)
  new_size = [int(s) for s in new_size]
  resample.SetSize(new_size)

  newimage = resample.Execute(data)
  return newimage

def rescale_prostate_to_CT(data, reference_img):
  """
  Resamples @data to new shape @new_shape with nearest neighhbor
  interpolation.
  """
  resample = sitk.ResampleImageFilter()
  resample.SetReferenceImage(reference_img)
  resample.SetInterpolator(sitk.sitkNearestNeighbor)

  return resample.Execute(data)

def one_hot(label):
  """
  Takes a binary tensor of shape (x, y, z) and returns
  tensor of shape (c, x, y, z) as one hot encoding.
  """
  label_oh = torch.zeros(2, label.shape[0], label.shape[1], label.shape[2])
  label_oh[0,...] = torch.where(label == 0, torch.tensor(1.0), torch.tensor(0.0))
  label_oh[1,...] = torch.where(label == 1, torch.tensor(1.0), torch.tensor(0.0))
  return label_oh

def one_hot_batch(label):
  """
  Takes a binary tensor of shape (n_batches, x, y, z)
  and returns a one hot encoded tensor of shape
  (n_batches, c, x, y, z)
  """
  label_oh = torch.zeros(label.shape[0], 2, label.shape[1], label.shape[2], label.shape[3])
  for b_idx, batch in enumerate(label):
    label_oh[b_idx, 0, ...] = torch.where(batch == 0, torch.tensor(1.0), torch.tensor(0.0))
    label_oh[b_idx, 1, ...] = torch.where(batch == 1, torch.tensor(1.0), torch.tensor(0.0))
  return label_oh

def tanh_batch(label):
  """
  Takes tensor with values 0 and 1 and returns tensor
  with values -1 and 1 for Tanh activation as final layer.
  """
  return torch.where(label == 0, torch.tensor(-1.), torch.tensor(1.))

def smooth_labels(label):
  """
  Takes label with only ones and zeros and returns uniformly
  distributed values between 0.0 and 0.2 instead of zeros
  and between 0.8 and 1.0 instead of ones.
  """
  zeros = torch.FloatTensor(label.shape).uniform_(0.0, 0.2)
  ones = torch.FloatTensor(label.shape).uniform_(0.8, 1.0)
  label = torch.where(label == 0, zeros, ones)
  return label

def get_centroid_coordinates(label_idx, label):
  """
  Find the centroid coordinates of a label map 'label' with
  label value 'label_idx'.
  """
  # Find outer borders of segmentation.
  nonzeros = np.argwhere(label == label_idx)
  maxX, maxY, maxZ = nonzeros[:, 0].max(), nonzeros[:, 1].max(), nonzeros[:, 2].max()
  minX, minY, minZ = nonzeros[:, 0].min(), nonzeros[:, 1].min(), nonzeros[:, 2].min()

  # Compute centroid.
  x = maxX - ((maxX - minX) // 2) if maxX != minX else minX
  y = maxY - ((maxX - minY) // 2) if maxY != minY else minY
  z = maxZ - ((maxX - minZ) // 2) if maxZ != minZ else minZ

  return (x, y, z)

# Logic or.
def tensor_or(t1, t2):
    return (t1 + t2) >= 1.0