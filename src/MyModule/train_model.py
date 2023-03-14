import torch
import torch.nn as nn
import torch.optim as optimizer

from CTGAN.ctgan.synthesizers import ctgan
from src.MyModule.preprocessing import StaticDataset


#%%

def train_gan(discriminator,
              generator,
              dataloader,
              epochs,
              lr,
              ) :

    optimD = optimizer.Adam(discriminator.parameters(), lr = lr)
    optimG = optimizer.Adam(generator.parameters(), lr = lr)

    schedulerD = optimizer.lr_scheduler.CosineAnnealingLR(optimD, 50, eta_min=0)
    schedulerG = optimizer.lr_scheduler.CosineAnnealingLR(optimG, 50, eta_min=0)

    criterion = nn.BCELoss()

    lossD_list = []
    lossG_list = []

    for i in range(epochs) :

        print(f"this is the {i}th epoch")
        batch_lossD = 0
        batch_lossG = 0
        
        for idx, batch in enumerate(dataloader) :

            ######################
            # Train Discriminator#
            ######################
            discriminator.zero_grad()
            true_label = torch.ones(len(batch)).reshape(-1,1)

            true_result = discriminator(batch)

            true_loss = criterion(true_result, true_label)
            true_loss.backward()

            generator.train()
            fake_label = torch.zeros(len(batch)).reshape(-1,1)

            noise = torch.rand(len(batch), batch.shape[1])
            generated = generator(noise)
            fake_result = discriminator(generated.detach())
            fake_loss = criterion(fake_result, fake_label)
            fake_loss.backward()

            lossD = fake_loss + true_loss
            optimD.step()

            ######################
            # Train Generator   ##
            ######################
            generator.zero_grad()

            output = discriminator(generated)
            lossG = criterion(output, true_label)
            lossG.backward()
            optimG.step()

            batch_lossD += lossD.item()
            batch_lossG += lossG.item()

        schedulerD.step()
        schedulerG.step()

        if i%10 == 0 :
            
            print(f"the loss for discriminator is {batch_lossD}")
            print(f"the loss for generator is {batch_lossG}")

        # append loss
        lossD_list.append(lossD.item())
        lossG_list.append(lossG.item())


    return generator, discriminator, lossD_list, lossG_list

def train_ctgan(dataset : StaticDataset,
                epochs,
                  ):

    data = dataset.processed_dataset
    model = ctgan.CTGAN(epochs = epochs)
    model.fit(data)
    return model







