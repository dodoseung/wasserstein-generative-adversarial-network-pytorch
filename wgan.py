import torch.nn.functional as F
import torch.nn as nn

# Wasserstein Generative Adversarial Network
class WGAN(nn.Module):
    def __init__(self, gen_latent_z=100, gen_init_layer=[7,7,64], gen_conv_trans=[2,2,1,1], gen_conv_filters=[128,64,64,1],
                 gen_conv_kernels=[5,5,5,5], gen_conv_strides=[1,1,1,1], gen_conv_pads=[2,2,2,2], gen_dropout_rate=0.1,
                 crt_input_img=[28,28,1], crt_conv_filters=[64,64,128,128], crt_conv_kernels=[5,5,5,5], 
                 crt_conv_strides=[2,2,2,1], crt_conv_pads=[2,2,2,2], crt_dropout_rate=0.4):
        super(WGAN, self).__init__()
        
        # Set the generator and discriminator
        self.G = Generator(gen_latent_z, gen_init_layer, gen_conv_trans, gen_conv_filters, gen_conv_kernels, gen_conv_strides, gen_conv_pads, gen_dropout_rate)
        self.C = Critic(crt_input_img, crt_conv_filters, crt_conv_kernels, crt_conv_strides, crt_conv_pads, crt_dropout_rate)

# Generator of GAN
class Generator(nn.Module):
    def __init__(self, latent_z, init_layer, conv_trans, conv_filters, conv_kernels, conv_strides, conv_pads, dropout_rate):
        super(Generator, self).__init__()
        # Initiation
        self.init_layer = init_layer
        self.latent_z = latent_z
        self.conv_trans = conv_trans
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.conv_pads = conv_pads
        self.dropout_rate = dropout_rate
        
        # Append filters for the init layer
        self.num_layers = len(self.conv_filters)
        self.conv_filters.insert(0, self.init_layer[2])
        
        # Fully connected layer
        self.fc = nn.Linear(self.latent_z, self.init_layer[0] * self.init_layer[1] * self.init_layer[2])
        self.dropout = nn.Dropout(p=self.dropout_rate)
    
        # Convolution layers
        self.conv = []
        for i in range(self.num_layers):
            layer = nn.Sequential(
                nn.Upsample(scale_factor=2),
                # nn.ConvTranspose2d(conv_filters[i], conv_filters[i], kernel_size=self.conv_trans[i], stride=self.conv_trans[i]),
                nn.Conv2d(conv_filters[i], conv_filters[i+1], kernel_size=self.conv_kernels[i], stride=self.conv_strides[i], padding=self.conv_pads[i]),
                nn.BatchNorm2d(conv_filters[i+1]),
                nn.ReLU())
            self.conv.append(layer)
        self.conv = nn.ModuleList(self.conv)
  
    def forward(self, z):
        # FCN
        x = F.relu(self.fc(z))
        
        # Reshape and drop out
        x = x.view(-1, self.init_layer[2], self.init_layer[0], self.init_layer[1])
        x = self.dropout(x)
        
        # CNN
        for i in range(self.num_layers):
            x = self.conv[i](x)
        
        return x
    
# Critic of GAN
class Critic(nn.Module):
    def __init__(self, input_img, conv_filters, conv_kernels, conv_strides, conv_pads, dropout_rate):
        super(Critic, self).__init__()
        # Initiation
        self.input_img = input_img
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.conv_pads = conv_pads
        self.dropout_rate = dropout_rate
        
        # Append filters for the input image
        self.num_layers = len(self.conv_filters)
        self.conv_filters.insert(0, self.input_img[2])
        
        # Convolution layers
        self.conv = []
        for i in range(self.num_layers):
            layer = nn.Sequential(
                nn.Conv2d(conv_filters[i], conv_filters[i+1], kernel_size=self.conv_kernels[i], stride=self.conv_strides[i], padding=self.conv_pads[i]),
                nn.BatchNorm2d(conv_filters[i+1]),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_rate))
            self.conv.append(layer)
        self.conv = nn.ModuleList(self.conv)
        
        # Output layer
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.conv_filters[-1], 1)
        
    def forward(self, x):
        # CNN
        for i in range(self.num_layers):
            x = self.conv[i](x)
            
        # Output layer
        x = self.pool(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc(x)
        
        return x


