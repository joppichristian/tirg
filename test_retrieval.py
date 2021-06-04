# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluates the retrieval model."""
import os
import numpy as np
import torch
from tqdm import tqdm as tqdm
import torch
import torch.utils.data
import torchvision
from tqdm import tqdm as tqdm
import argparse
import datasets
import img_text_composition_models
from matplotlib import pyplot as plt
def parse_opt():
  """Parses the input arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('-f', type=str, default='')
  parser.add_argument('--comment', type=str, default='Fashion200K_Eval')
  parser.add_argument('--dataset', type=str, default='fashion200k')
  parser.add_argument(
      '--dataset_path', type=str, default='dataset/Fashion200K')
  parser.add_argument('--model', type=str, default='tirg')
  parser.add_argument('--embed_dim', type=int, default=512)
  parser.add_argument('--batch_size', type=int, default=32)
  parser.add_argument('--loader_num_workers', type=int, default=0)
  args = parser.parse_args()
  return args


def load_dataset(opt):
  """Loads the input datasets."""
  print('Reading dataset ', opt.dataset)
  if opt.dataset == 'css3d':
    trainset = datasets.CSSDataset(
        path=opt.dataset_path,
        split='train',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ]))
    testset = datasets.CSSDataset(
        path=opt.dataset_path,
        split='test',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ]))
  elif opt.dataset == 'fashion200k':
    trainset = datasets.Fashion200k(
        path=opt.dataset_path,
        split='train',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ]))
    testset = datasets.Fashion200k(
        path=opt.dataset_path,
        split='test',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ]))
  elif opt.dataset == 'mitstates':
    trainset = datasets.MITStates(
        path=opt.dataset_path,
        split='train',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ]))
    testset = datasets.MITStates(
        path=opt.dataset_path,
        split='test',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ]))
  else:
    print('Invalid dataset', opt.dataset)
    sys.exit()

  print('trainset size:', len(trainset))
  print('testset size:', len(testset))
  return trainset, testset


def create_model_and_optimizer(opt, texts):
  """Builds the model and related optimizer."""
  print('Creating model and optimizer for', opt.model)
  if opt.model == 'imgonly':
    model = img_text_composition_models.SimpleModelImageOnly(
        texts, embed_dim=opt.embed_dim)
  elif opt.model == 'textonly':
    model = img_text_composition_models.SimpleModelTextOnly(
        texts, embed_dim=opt.embed_dim)
  elif opt.model == 'concat':
    model = img_text_composition_models.Concat(texts, embed_dim=opt.embed_dim)
  elif opt.model == 'tirg':
    model = img_text_composition_models.TIRG(texts, embed_dim=opt.embed_dim)
  elif opt.model == 'tirg_lastconv':
    model = img_text_composition_models.TIRGLastConv(
        texts, embed_dim=opt.embed_dim)
  else:
    print('Invalid model', opt.model)
    print('available: imgonly, textonly, concat, tirg or tirg_lastconv')
    sys.exit()
  model = model.cuda()

  return model


def test(opt, model, testset):
  """Tests a model over the given testset."""
  model.eval()
  test_queries = testset.get_test_queries()
  all_imgs = []
  all_captions = []
  all_queries = []
  all_target_captions = []
  if test_queries:
    # compute test query features
    imgs = []
    mods = []
    for t in tqdm(test_queries):
      imgs += [testset.get_img(t['source_img_id'])]
      mods += [t['mod']['str']]
      if len(imgs) >= opt.batch_size or t is test_queries[-1]:
        if 'torch' not in str(type(imgs[0])):
          imgs = [torch.from_numpy(d).float() for d in imgs]
        imgs = torch.stack(imgs).float()
        imgs = torch.autograd.Variable(imgs).cuda()
        f = model.compose_img_text(imgs, mods).data.cpu().numpy()
        all_queries += [f]
        imgs = []
        mods = []
    all_queries = np.concatenate(all_queries)
    all_target_captions = [t['target_caption'] for t in test_queries]

    # compute all image features
    imgs = []
    for i in tqdm(range(len(testset.imgs))):
      imgs += [testset.get_img(i)]
      if len(imgs) >= opt.batch_size or i == len(testset.imgs) - 1:
        if 'torch' not in str(type(imgs[0])):
          imgs = [torch.from_numpy(d).float() for d in imgs]
        imgs = torch.stack(imgs).float()
        imgs = torch.autograd.Variable(imgs).cuda()
        imgs = model.extract_img_feature(imgs).data.cpu().numpy()
        all_imgs += [imgs]
        imgs = []
    all_imgs = np.concatenate(all_imgs)
    all_captions = [img['captions'][0] for img in testset.imgs]

  # feature normalization
  for i in range(all_queries.shape[0]):
    all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
  for i in range(all_imgs.shape[0]):
    all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])

  # match test queries to target images, get nearest neighbors
  nn_result = []
  result_per_query = np.zeros((all_queries.shape[0],))
  caption_dict = {x:ii for ii,x in enumerate(all_captions)}
  
  for i in tqdm(range(all_queries.shape[0])):
    sims = all_queries[i:(i+1), :].dot(all_imgs.T)
    if test_queries:
      sims[0, test_queries[i]['source_img_id']] = -10e10  # remove query image
    nn_result.append(np.argsort(-sims[0, :])[:110])
    capt_query_nn = [all_captions[xx] for xx in np.argsort(-sims[0, :])]
    capt_query_nn = np.array([caption_dict[xx] for xx in capt_query_nn])
    capt_target = caption_dict[all_target_captions[i]]
    result_per_query[i] = np.where(capt_query_nn == capt_target)[0][0]
  nn_backup = nn_result.copy()
  # compute recalls
  out = []
  
  nn_result = [[all_captions[nn] for nn in nns[:110]] for nns in nn_result]
  
  for k in [1, 5, 10, 50, 100]:
    r = 0.0
    for i, nns in enumerate(nn_result):
      if all_target_captions[i] in nns[:k]:
        r += 1
    r /= len(nn_result)
    
    out += [('recall_top' + str(k) + '_correct_composition', r)]

    if opt.dataset == 'mitstates':
      r = 0.0
      for i, nns in enumerate(nn_result):
        if all_target_captions[i].split()[0] in [c.split()[0] for c in nns[:k]]:
          r += 1
      r /= len(nn_result)
      out += [('recall_top' + str(k) + '_correct_adj', r)]

      r = 0.0
      for i, nns in enumerate(nn_result):
        if all_target_captions[i].split()[1] in [c.split()[1] for c in nns[:k]]:
          r += 1
      r /= len(nn_result)
      out += [('recall_top' + str(k) + '_correct_noun', r)]

  # nn_backup = torch.load('backup.pth')
  
  ordered_indexes = result_per_query.argsort()
  for ii in ordered_indexes[:100]:
    fig, axs = plt.subplots(1,6)
    t = test_queries[ii]
    img = (testset.get_img(t['source_img_id']).permute(1,2,0) * torch.tensor([0.229, 0.224, 0.225])) + torch.tensor([0.485, 0.456, 0.406]) 
    axs[0].imshow((img.numpy()*255.0).astype(np.uint8))
    cap = "\n".join(all_captions[t['source_img_id']].split(" "))
    axs[0].set_title('Query\n' + cap, fontsize=8)
    axs[0].axis('off')
    nns = nn_backup[ii][:5]
    for jj,nn in enumerate(nns):
      img_target = (testset.get_img(nn).permute(1,2,0) * torch.tensor([0.229, 0.224, 0.225])) + torch.tensor([0.485, 0.456, 0.406]) 
      axs[jj+1].imshow((img_target.numpy()*255.0).astype(np.uint8))
      axs[jj+1].axis('off')
      if result_per_query[ii] == jj:
        cap = "\n".join(all_captions[nn].split(" "))
        axs[jj+1].set_title(f'R@{jj+1} - Yes!\n{cap}', fontsize=8)
      else:
        cap = "\n".join(all_captions[nn].split(" "))
        axs[jj+1].set_title(f'R@{jj+1} - No!\n{cap}', fontsize=8)
    plt.suptitle(f"Pos: {int(result_per_query[ii]+1)} - Modify: {t['mod']['str']}")
    os.makedirs(os.path.join('results', 'best'), exist_ok=True)
    plt.savefig(os.path.join('results', 'best', str(ii) + '.png'))
    plt.close()

  ordered_indexes = result_per_query.argsort()[::-1]
  for ii in ordered_indexes[:100]:
    fig, axs = plt.subplots(1,6)
    t = test_queries[ii]
    img = (testset.get_img(t['source_img_id']).permute(1,2,0) * torch.tensor([0.229, 0.224, 0.225])) + torch.tensor([0.485, 0.456, 0.406]) 
    axs[0].imshow((img.numpy()*255.0).astype(np.uint8))
    cap = "\n".join(all_captions[t['source_img_id']].split(" "))
    axs[0].set_title('Query\n'+cap, fontsize=8)
    axs[0].axis('off')
    nns = nn_backup[ii][:5]
    for jj,nn in enumerate(nns):
      img_target = (testset.get_img(nn).permute(1,2,0) * torch.tensor([0.229, 0.224, 0.225])) + torch.tensor([0.485, 0.456, 0.406])
                                             
      axs[jj+1].imshow((img_target.numpy()*255.0).astype(np.uint8))
      axs[jj+1].axis('off')
      if result_per_query[ii] == jj:
        cap = "\n".join(all_captions[nn].split(" "))
        axs[jj+1].set_title(f'R@{jj+1} - Yes!\n{cap}', fontsize=8)
      else:
        cap = "\n".join(all_captions[nn].split(" "))
        axs[jj+1].set_title(f'R@{jj+1} - No!\n{cap}', fontsize=8)
    plt.suptitle(f"Pos: {int(result_per_query[ii]+1)} - Modify: {t['mod']['str']}")
    os.makedirs(os.path.join('results', 'worst'), exist_ok=True)
    plt.savefig(os.path.join('results', 'worst', str(ii) + '.png'))
    plt.close()

  return out


if __name__ == '__main__':
  opt = parse_opt()
  print('Arguments:')
  for k in opt.__dict__.keys():
    print('    ', k, ':', str(opt.__dict__[k]))

  trainset, testset = load_dataset(opt)
  model = create_model_and_optimizer(opt, trainset.get_all_texts())
  tmp = torch.load('pre_trained/checkpoint_fashion200k.pth')
  model.load_state_dict(tmp['model_state_dict'])
  print(test(opt, model, testset))
