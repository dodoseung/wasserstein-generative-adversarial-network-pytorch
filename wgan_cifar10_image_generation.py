from wgan import WGAN

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

import numpy as np
from utils import save_model, load_yaml

# Set the configuration
config = load_yaml("./config/wgan_cifar10_config.yml")

# Training setting
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(config['data']['seed'])
if device == 'cuda':
  torch.cuda.manual_seed_all(config['data']['seed'])

# Set the transform
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize(config['data']['img_size'])])

# Set the training data
train_data = datasets.CIFAR10(config['data']['data_path'], download=config['data']['download'], train=True, transform=transform)
# Split the horse data
train_data = torch.utils.data.Subset(train_data, np.where(np.array(train_data.targets) == 7)[0])
train_loader = torch.utils.data.DataLoader(train_data, batch_size=config['data']['batch_size'], shuffle=config['data']['shuffle'], drop_last=config['data']['drop_last'])

# Set the model
model = WGAN(gen_latent_z=config['model']['gen_latent_z'], gen_init_layer=config['model']['gen_init_layer'],
            gen_conv_trans=config['model']['gen_conv_trans'], gen_conv_filters=config['model']['gen_conv_filters'],
            gen_conv_kernels=config['model']['gen_conv_kernels'], gen_conv_strides=config['model']['gen_conv_strides'],
            gen_conv_pads=config['model']['gen_conv_pads'],gen_dropout_rate=config['model']['gen_dropout_rate'],
            crt_input_img=config['model']['crt_input_img'], crt_conv_filters=config['model']['crt_conv_filters'],
            crt_conv_kernels=config['model']['crt_conv_kernels'], crt_conv_strides=config['model']['crt_conv_strides'],
            crt_conv_pads=config['model']['crt_conv_pads'], crt_dropout_rate=config['model']['crt_dropout_rate']).to(device)

print(model, device)

# Set the criterion and optimizer
g_optimizer = optim.RMSprop(model.G.parameters(), lr=config['train']['lr'])
c_optimizer = optim.RMSprop(model.C.parameters(), lr=config['train']['lr'])
criterion = nn.BCELoss()

# Set values
batch_size = config['data']['batch_size']
z_latent = config['model']['gen_latent_z']
gen_iteration = config ['train']['gen_iteration']
crt_clip_value = config['train']['crt_clip_value']

# Training
def train(epoch, train_loader, g_optimizer, c_optimizer):
  model.train()
  g_train_loss = 0.0
  g_train_num = 0
  c_train_loss = 0.0
  c_train_num = 0
  
  for i, data in enumerate(train_loader, 0):
    # Critic
    # get the inputs; data is a list of [inputs, labels]
    real_img, _ = data

    # Transfer data to device
    real_img = real_img.to(device)
    real_score = model.C(real_img)

    # Generate generated image
    z = 2 * torch.rand(batch_size, z_latent, device=device) - 1
    fake_img = model.G(z)
    fake_score = model.C(fake_img)
    
    # Loss for the critic with EM distance
    c_loss = fake_score.mean() - real_score.mean()
    
    # Training for the critic
    c_optimizer.zero_grad()
    c_loss.backward()
    c_optimizer.step()
    
    # Clip weights of discriminator
    for p in model.C.parameters():
      p.data.clamp_(-crt_clip_value, crt_clip_value)
    
    # Generator
    if i % gen_iteration == 0:
      # Get the fake images and scores
      z = 2 * torch.rand(batch_size, z_latent, device=device) - 1
      fake_img = model.G(z)
      fake_score = model.C(fake_img)

      # Training for the generator
      g_loss = - fake_score.mean()
      g_optimizer.zero_grad()
      g_loss.backward()
      g_optimizer.step()
      
      # loss
      g_train_loss += g_loss.item()
      g_train_num += fake_img.size(0)

    # loss
    c_train_loss += c_loss.item()
    c_train_num += real_img.size(0)
    
    if i % config['others']['log_period'] == 0 and i != 0:
      print(f'[{epoch}, {i}]\t Train loss: (G){g_train_loss / g_train_num:.10f}, (D){c_train_loss / c_train_num:.10f}')
  
  # Average loss
  c_train_loss /= c_train_num
  
  return c_train_loss

# Main
if __name__ == '__main__':
  for epoch in range(config['train']['epochs']):  # loop over the dataset multiple times
    # Training
    train_loss = train(epoch, train_loader, g_optimizer, c_optimizer)
    
    # Print the log
    print(f'Epoch: {epoch}\t Train loss: {train_loss:.10f}')
    
    # Save the model
    save_model(model_name=config['save']['model_name'], epoch=epoch, model=model, optimizer=c_optimizer, loss=train_loss, config=config)
    