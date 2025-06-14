import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

# Generator: U-Net Architecture
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNetGenerator, self).__init__()
        
        def down_conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True)
            )
        
        def up_conv(in_c, out_c):
            return nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5)
            )
        
        # Encoder
        self.down1 = down_conv(in_channels, 64)
        self.down2 = down_conv(64, 128)
        self.down3 = down_conv(128, 256)
        self.down4 = down_conv(256, 512)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.up1 = up_conv(512, 512)
        self.up2 = up_conv(1024, 256)
        self.up3 = up_conv(512, 128)
        self.up4 = up_conv(256, 64)
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        
        # Bottleneck
        b = self.bottleneck(d4)
        
        # Decoder with skip connections
        u1 = self.up1(b)
        u2 = self.up2(torch.cat([u1, d4], dim=1))
        u3 = self.up3(torch.cat([u2, d3], dim=1))
        u4 = self.up4(torch.cat([u3, d2], dim=1))
        return self.final(torch.cat([u4, d1], dim=1))

# Discriminator: PatchGAN
class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=6):
        super(PatchGANDiscriminator, self).__init__()
        
        def discriminator_block(in_c, out_c, stride=2):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True)
            )
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            discriminator_block(64, 128),
            discriminator_block(128, 256),
            discriminator_block(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))

# Custom Dataset
class PairedImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_pairs = []
        
        # Assuming paired images are in format: input_image.jpg, target_image.jpg
        for img_name in os.listdir(root_dir):
            if img_name.endswith('_input.jpg'):
                input_path = os.path.join(root_dir, img_name)
                target_path = os.path.join(root_dir, img_name.replace('_input.jpg', '_target.jpg'))
                if os.path.exists(target_path):
                    self.image_pairs.append((input_path, target_path))
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        input_path, target_path = self.image_pairs[idx]
        input_img = Image.open(input_path).convert('RGB')
        target_img = Image.open(target_path).convert('RGB')
        
        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)
        
        return input_img, target_img

# Training Function
def train_pix2pix(dataset_path, epochs=100, batch_size=1, lr=0.0002):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Dataset and DataLoader
    dataset = PairedImageDataset(dataset_path, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Models
    generator = UNetGenerator().to(device)
    discriminator = PatchGANDiscriminator().to(device)
    
    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # Loss functions
    gan_loss = nn.BCELoss()
    l1_loss = nn.L1Loss()
    
    # Training loop
    for epoch in range(epochs):
        for i, (input_imgs, target_imgs) in enumerate(dataloader):
            input_imgs, target_imgs = input_imgs.to(device), target_imgs.to(device)
            
            # Train Discriminator
            d_optimizer.zero_grad()
            
            # Real pair
            real_pair = discriminator(input_imgs, target_imgs)
            real_loss = gan_loss(real_pair, torch.ones_like(real_pair))
            
            # Fake pair
            fake_imgs = generator(input_imgs)
            fake_pair = discriminator(input_imgs, fake_imgs.detach())
            fake_loss = gan_loss(fake_pair, torch.zeros_like(fake_pair))
            
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            d_optimizer.step()
            
            # Train Generator
            g_optimizer.zero_grad()
            fake_pair = discriminator(input_imgs, fake_imgs)
            g_gan_loss = gan_loss(fake_pair, torch.ones_like(fake_pair))
            g_l1_loss = l1_loss(fake_imgs, target_imgs) * 100  # L1 loss weight
            g_loss = g_gan_loss + g_l1_loss
            g_loss.backward()
            g_optimizer.step()
            
            if i % 100 == 0:
                print(f"Epoch [{epoch}/{epochs}] Batch [{i}/{len(dataloader)}] "
                      f"D Loss: {d_loss.item():.4f} G Loss: {g_loss.item():.4f}")
    
    # Save models
    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')

if __name__ == '__main__':
    # Example usage
    dataset_path = 'path/to/your/paired/image/dataset'
    train_pix2pix(dataset_path)