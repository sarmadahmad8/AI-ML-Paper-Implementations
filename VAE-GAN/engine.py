import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from model import VAEGAN, Discriminator, init_weights

def kl_divergence_loss(mean: torch.Tensor,
                       logvar: torch.Tensor) -> torch.Tensor:
    Dkl_loss = 1 + logvar - mean.pow(2) - logvar.exp()
    Dkl_loss = (-0.5 * torch.sum(Dkl_loss))/torch.numel(mean.data)
    return Dkl_loss

lr = 1e-4
epochs = 25
gamma = 5.0

device = "cuda" if torch.cuda.is_available() else "cpu"

vae_gan = VAEGAN(in_channels=3,
                 latent_dim=128,
                 out_channels=3).to(device=device)

loss_fn = nn.BCELoss()
encoder_optim = torch.optim.RMSprop(vae_gan.encoder.parameters(),
                                    lr = lr)
decoder_optim = torch.optim.RMSprop(vae_gan.decoder.parameters(),
                                    lr = lr)
discriminator_optim = torch.optim.RMSprop(vae_gan.discriminator.parameters(),
                                    lr= lr)

batch_losses = {"encoder_loss": [],
                "decoder_loss": [],
                "discriminator_loss": [],
                "cosine_similarity_metric": [],
                "mse_metric": []}

test_batch, _ = next(iter(test_dataloader))
test_batch = test_batch.to(device)

for epoch in tqdm(range(epochs)):
    
    total_encoder_loss, total_decoder_loss, total_discriminator_loss = 0.0, 0.0, 0.0
    total_prior_loss, total_Disl_llike_loss, total_gan_loss = 0.0, 0.0, 0.0
    total_cosine_similarity, total_mse = 0.0, 0.0
    
    for batch, (X, _) in tqdm(enumerate(train_dataloader)):
        
        vae_gan.train()
        X = X.to(device)
        
        z_mean, z_logvar, _, x_tilda = vae_gan(X)

        z_p = torch.randn_like(z_mean).to(device)
        x_p = vae_gan.decoder(z_p)

        _, disc_x_p = vae_gan.discriminator(x_p)
        discl_x_tilda, disc_x_tilda  = vae_gan.discriminator(x_tilda)
        discl_x, disc_x = vae_gan.discriminator(X)

        ones_label = torch.ones_like(disc_x, requires_grad=False).to(device)
        zeros_label = torch.zeros_like(disc_x, requires_grad=False).to(device)
        
        gan_loss = loss_fn(disc_x, ones_label) + loss_fn(disc_x_tilda,zeros_label) + loss_fn(disc_x_p, zeros_label)
        discriminator_loss = gan_loss
        
        discriminator_optim.zero_grad()
        gan_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(vae_gan.discriminator.parameters(), max_norm=1.0)
        discriminator_optim.step()

        discl_x_tilda, disc_x_tilda  = vae_gan.discriminator(x_tilda)
        discl_x, disc_x = vae_gan.discriminator(X)

        Disl_llike_loss = ((discl_x_tilda - discl_x) ** 2).mean()

        z_p = torch.randn_like(z_mean).to(device)
        x_p = vae_gan.decoder(z_p)

        _, disc_x_p = vae_gan.discriminator(x_p)
        gan_loss = loss_fn(disc_x, ones_label) + loss_fn(disc_x_tilda,zeros_label) + loss_fn(disc_x_p, zeros_label)
        decoder_loss = gamma * Disl_llike_loss - gan_loss

        decoder_optim.zero_grad()
        decoder_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(vae_gan.decoder.parameters(), max_norm=1.0)
        decoder_optim.step()
        
        z_mean, z_logvar, _, x_tilda = vae_gan(X)
        prior_loss = kl_divergence_loss(mean= z_mean,
                                      logvar= z_logvar)

        discl_x_tilda, disc_x_tilda  = vae_gan.discriminator(x_tilda)
        discl_x, disc_x = vae_gan.discriminator(X)

        Disl_llike_loss = ((discl_x_tilda - discl_x) ** 2).mean()
        
        encoder_loss = prior_loss + Disl_llike_loss
        
        encoder_optim.zero_grad()
        encoder_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(vae_gan.encoder.parameters(), max_norm=1.0)
        encoder_optim.step()

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
            print(f" Prior Loss: {total_prior_loss/batch:.5f} | Discriminator Layer Loss: {total_Disl_llike_loss/batch:.5f} | Gan Loss: {total_gan_loss/batch:.5f}")
            print(f" Cosine Similarity: {cosine_similarity/batch:.5f} | Mean Squared Error: {mse/batch:.5f}")
            #  print(f" Encoder Loss: {total_encoder_loss/batch:.5f} | Decoder Loss: {total_decoder_loss/batch:.5f} | Discriminator Loss: {total_discriminator_loss/batch:.5f} ")
            vae_gan.eval()
            with torch.inference_mode():
                _, _, _, generated_image_batch = vae_gan(test_batch)
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

    print(f" Epoch: {epoch+1} | Average Encoder Loss: {total_encoder_loss:.5f} | Average Decoder Loss: {total_decoder_loss:.5f} | Average Discriminator Loss: {total_discriminator_loss:.5f}")
    print(f" Epoch: {epoch+1} | Average Cosine Similarity: {total_cosine_similarity:.5f} | Average Mean Squared Error: {total_mse:.5f}")
    
    batch_losses["encoder_loss"].append(total_encoder_loss)
    batch_losses["decoder_loss"].append(total_decoder_loss)
    batch_losses["discriminator_loss"].append(total_discriminator_loss)
    batch_losses["cosine_similarity_metric"].append(total_cosine_similarity)
    batch_losses["mse_metric"].append(total_mse)
    
