import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from model import VAEGAN, Discriminator, init_weights
from utils import plot_image_grid

class Losses:
        
    def prior_loss(self,
                   mean: torch.Tensor,
                   logvar: torch.Tensor) -> torch.Tensor:
        
        Dkl_loss = 1 + logvar - mean.pow(2) - logvar.exp()
        Dkl_loss = (-0.5 * torch.sum(Dkl_loss))/torch.numel(mean.data)
        return Dkl_loss
    
    def disl_llike_loss(self,
                        discl_x: torch.Tensor,
                        discl_x_tilda: torch.Tensor) -> torch.Tensor:
    
        disl_llike_loss = ((discl_x_tilda - discl_x) ** 2).mean()
    
        return disl_llike_loss
    
    def gan_loss(self,
                 disc_x: torch.Tensor,
                 disc_x_tilda: torch.Tensor,
                 disc_x_p: torch.Tensor) -> torch.Tensor:

        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        loss_fn = nn.BCELoss()
    
        ones_label =  torch.ones_like(disc_x, requires_grad= False).to(device)
        zeros_label = torch.zeros_like(disc_x, requires_grad= False).to(device)
    
        gan_loss = loss_fn(disc_x, ones_label) + loss_fn(disc_x_tilda, zeros_label) + loss_fn(disc_x_p, zeros_label)
    
        return gan_loss

def train_step(model: torch.nn.Module,
               train_dataloader: torch.utils.data.DataLoader,
               losses: Losses,
               discriminator_optimizer: torch.optim.Optimizer,
               decoder_optimizer: torch.optim.Optimizer,
               encoder_optimizer: torch.optim.Optimizer,
               test_batch: torch.utils.data.DataLoader,
               device: torch.device = "cuda",
               gamma: float = 1.0):
    
    total_encoder_loss, total_decoder_loss, total_discriminator_loss = 0.0, 0.0, 0.0
    total_prior_loss, total_Disl_llike_loss, total_gan_loss = 0.0, 0.0, 0.0
    total_cosine_similarity, total_mse = 0.0, 0.0
    
    for batch, (X, _) in tqdm(enumerate(train_dataloader)):
        
        model.train()
        X = X.to(device)
        
        z_mean, z_logvar, _, x_tilda = model(X)

        z_p = torch.randn_like(z_mean).to(device)
        x_p = model.decoder(z_p)

        _, disc_x_p = model.discriminator(x_p)
        discl_x_tilda, disc_x_tilda  = model.discriminator(x_tilda)
        discl_x, disc_x = model.discriminator(X)

        gan_loss = losses.gan_loss(disc_x= disc_x,
                                   disc_x_p= disc_x_p,
                                   disc_x_tilda= disc_x_tilda)

        discriminator_loss = gan_loss
        
        discriminator_optimizer.zero_grad()
        discriminator_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), max_norm=1.0)
        discriminator_optimizer.step()

        discl_x_tilda, disc_x_tilda  = model.discriminator(x_tilda)
        discl_x, disc_x = model.discriminator(X)

        Disl_llike_loss = losses.disl_llike_loss(discl_x_tilda= discl_x_tilda,
                                                 discl_x= discl_x)

        z_p = torch.randn_like(z_mean).to(device)
        x_p = model.decoder(z_p)

        _, disc_x_p = model.discriminator(x_p)
        
        gan_loss = losses.gan_loss(disc_x= disc_x,
                                   disc_x_p= disc_x_p,
                                   disc_x_tilda= disc_x_tilda)
        
        decoder_loss = gamma * Disl_llike_loss - gan_loss

        decoder_optimizer.zero_grad()
        decoder_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=1.0)
        decoder_optimizer.step()
        
        z_mean, z_logvar, _, x_tilda = model(X)
        prior_loss = losses.prior_loss(mean= z_mean,
                                      logvar= z_logvar)

        discl_x_tilda, disc_x_tilda  = model.discriminator(x_tilda)
        discl_x, disc_x = model.discriminator(X)

        Disl_llike_loss = losses.disl_llike_loss(discl_x_tilda= discl_x_tilda,
                                                 discl_x= discl_x)
        
        encoder_loss = prior_loss + Disl_llike_loss
        
        encoder_optimizer.zero_grad()
        encoder_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), max_norm=1.0)
        encoder_optimizer.step()

        cosine_similarity = F.cosine_similarity(x1= x_tilda.flatten(start_dim=1, end_dim=-1),
                                                x2= X.flatten(start_dim=1, end_dim=-1),
                                                dim= 1).mean()
        mse = F.mse_loss(input= x_tilda,
                         target= X)
        
        total_decoder_loss+= decoder_loss.item()
        total_encoder_loss+= encoder_loss.item()
        total_discriminator_loss += discriminator_loss.item()
        total_prior_loss += prior_loss.item()
        total_Disl_llike_loss += Disl_llike_loss.item()
        total_gan_loss += gan_loss.item()
        total_cosine_similarity += cosine_similarity.item()
        total_mse += mse.item()


        if batch % 200 == 0:
            batch = batch+1
            print(f"Prior Loss: {total_prior_loss/batch:.5f} | Discriminator Layer Loss: {total_Disl_llike_loss/batch:.5f} | Gan Loss: {total_gan_loss/batch:.5f}")
            print(f"Cosine Similarity: {total_cosine_similarity/batch:.5f} | Mean Squared Error: {total_mse/batch:.5f}")
            #  print(f" Encoder Loss: {total_encoder_loss/batch:.5f} | Decoder Loss: {total_decoder_loss/batch:.5f} | Discriminator Loss: {total_discriminator_loss/batch:.5f} ")
            model.eval()
            with torch.inference_mode():
                _, _, _, generated_image_batch = model(test_batch)
                generated_image_sample = generated_image_batch[0]
                generated_image_sample = ((generated_image_sample + 1) / 2)
                plt.imshow(generated_image_sample.permute(1, 2, 0).detach().cpu())
                plt.axis(False)
                plt.show()
                
    total_encoder_loss /= len(train_dataloader)
    total_decoder_loss /= len(train_dataloader)
    total_discriminator_loss /= len(train_dataloader)
    total_cosine_similarity /= len(train_dataloader)
    total_mse /= len(train_dataloader)

    print(f"Average Encoder Loss: {total_encoder_loss:.5f} | Average Decoder Loss: {total_decoder_loss:.5f} | Average Discriminator Loss: {total_discriminator_loss:.5f}")
    print(f"Average Cosine Similarity: {total_cosine_similarity:.5f} | Average Mean Squared Error: {total_mse:.5f}")

    return total_encoder_loss, total_decoder_loss, total_discriminator_loss, total_cosine_similarity, total_mse

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              losses: Losses,
              device: torch.device = "cuda",
              gamma: float = 1.0):
    
    total_encoder_loss, total_decoder_loss, total_discriminator_loss = 0.0, 0.0, 0.0
    total_prior_loss, total_Disl_llike_loss, total_gan_loss = 0.0, 0.0, 0.0
    total_cosine_similarity, total_mse = 0.0, 0.0
    
    model.eval()

    with torch.inference_mode():
        for batch, (X, _) in tqdm(enumerate(dataloader)):
            X = X.to(device)
            
            z_mean, z_logvar, _, x_tilda = model(X)
        
            z_p = torch.randn_like(z_mean).to(device)
            x_p = model.decoder(z_p)
        
            _, disc_x_p = model.discriminator(x_p)
            discl_x_tilda, disc_x_tilda  = model.discriminator(x_tilda)
            discl_x, disc_x = model.discriminator(X)
        
            gan_loss = losses.gan_loss(disc_x= disc_x,
                                   disc_x_p= disc_x_p,
                                   disc_x_tilda= disc_x_tilda)
            
            discriminator_loss = gan_loss
        
            discl_x_tilda, disc_x_tilda  = model.discriminator(x_tilda)
            discl_x, disc_x = model.discriminator(X)
        
            Disl_llike_loss = losses.disl_llike_loss(discl_x_tilda= discl_x_tilda,
                                                     discl_x= discl_x)
        
            decoder_loss = gamma * Disl_llike_loss - gan_loss
    
            prior_loss = losses.prior_loss(mean= z_mean,
                                          logvar= z_logvar)
            
            encoder_loss = prior_loss + Disl_llike_loss
        
            cosine_similarity = F.cosine_similarity(x1= x_tilda.flatten(start_dim=1, end_dim=-1),
                                                    x2= X.flatten(start_dim=1, end_dim=-1),
                                                    dim= 1).mean()
            mse = F.mse_loss(input= x_tilda,
                             target= X)
            
            total_decoder_loss+= decoder_loss.item()
            total_encoder_loss+= encoder_loss.item()
            total_discriminator_loss += discriminator_loss.item()
            total_prior_loss += prior_loss.item()
            total_Disl_llike_loss += Disl_llike_loss.item()
            total_gan_loss += gan_loss.item()
            total_cosine_similarity += cosine_similarity.item()
            total_mse += mse.item()


        total_encoder_loss /= len(dataloader)
        total_decoder_loss /= len(dataloader)
        total_discriminator_loss /= len(dataloader)
        total_cosine_similarity /= len(dataloader)
        total_mse /= len(dataloader)
        
        print(f"Average Encoder Loss: {total_encoder_loss:.5f} | Average Decoder Loss: {total_decoder_loss:.5f} | Average Discriminator Loss: {total_discriminator_loss:.5f}")
        print(f"Average Cosine Similarity: {total_cosine_similarity:.5f} | Average Mean Squared Error: {total_mse:.5f}")

        plot_image_grid(model = model,
                        dataloader= dataloader,
                        grid_dim= 8, 
                        device = device)

    return total_encoder_loss, total_decoder_loss, total_discriminator_loss, total_cosine_similarity, total_mse

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          lr: float = 1e-4,
          epochs: int = 25,
          gamma: float = 5.0,
          alpha: float = 0.1,
          device: torch.device = "cuda"):

    lr = lr
    epochs = epochs
    gamma = gamma
    
    model.to(device)
    
    losses = Losses()
    
    encoder_optimizer = torch.optim.RMSprop(model.encoder.parameters(),
                                        lr = lr)
    decoder_optimizer = torch.optim.RMSprop(model.decoder.parameters(),
                                        lr = lr)
    discriminator_optimizer = torch.optim.RMSprop(model.discriminator.parameters(),
                                        lr= lr * alpha)
    
    train_batch_losses = {"encoder_loss": [],
                    "decoder_loss": [],
                    "discriminator_loss": [],
                    "cosine_similarity_metric": [],
                    "mse_metric": []}

    test_batch_losses = {"encoder_loss": [],
                    "decoder_loss": [],
                    "discriminator_loss": [],
                    "cosine_similarity_metric": [],
                    "mse_metric": []}
    
    test_batch, _ = next(iter(test_dataloader))
    test_batch = test_batch.to(device)
    
    for epoch in tqdm(range(epochs)):
        total_encoder_loss, total_decoder_loss, total_discriminator_loss, total_cosine_similarity, total_mse = train_step(model= model,
                                                                                                                           train_dataloader= train_dataloader,
                                                                                                                           losses= losses,
                                                                                                                           discriminator_optimizer= discriminator_optimizer,
                                                                                                                           decoder_optimizer= decoder_optimizer,
                                                                                                                           encoder_optimizer= encoder_optimizer,
                                                                                                                           test_batch= test_batch,
                                                                                                                           device= device,
                                                                                                                           gamma = gamma)

        train_batch_losses["encoder_loss"].append(total_encoder_loss)
        train_batch_losses["decoder_loss"].append(total_decoder_loss)
        train_batch_losses["discriminator_loss"].append(total_discriminator_loss)
        train_batch_losses["cosine_similarity_metric"].append(total_cosine_similarity)
        train_batch_losses["mse_metric"].append(total_mse)

        total_encoder_loss, total_decoder_loss, total_discriminator_loss, total_cosine_similarity, total_mse = test_step(model= model,
                                                                                                                          dataloader= test_dataloader,
                                                                                                                          losses= losses,
                                                                                                                          device= device,
                                                                                                                          gamma = gamma)

        test_batch_losses["encoder_loss"].append(total_encoder_loss)
        test_batch_losses["decoder_loss"].append(total_decoder_loss)
        test_batch_losses["discriminator_loss"].append(total_discriminator_loss)
        test_batch_losses["cosine_similarity_metric"].append(total_cosine_similarity)
        test_batch_losses["mse_metric"].append(total_mse)

    return train_batch_losses, test_batch_losses
