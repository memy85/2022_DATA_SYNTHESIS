
#%%

import pandas as pd
import numpy as np
import torch


class StaticRestorer :

    def __init__(self, generator, static_dataset, num_to_generate) : 
        self.generator = generator
        self.static_dataset = static_dataset
        self.num_to_generate = num_to_generate 
        self.feature_size = self.generator.input_size
        
        self.generate_data()


    def generate_data(self) : 
        noise = torch.rand(self.num_to_generate, self.feature_size)
        batch_size = 128
        max_batch = self.feature_size // batch_size
        last_batch_size = self.feature_size % batch_size

        generated_output = []
        for batch_idx in range(max_batch + 1) : 
            if batch_idx == max_batch :
                noise = torch.rand(last_batch_size, self.feature_size)
            else :
                noise = torch.rand(self.batch_size, self.feature_size)

            with torch.no_grad() :
                output = self.generator(noise)
            generated_output.append(output.numpy())

        self.raw_output = np.concatenate(generated_output, axis = 0)

    def restore_to_static(self) :
        df = pd.DataFrame(self.raw_output)
        id_column = pd.DataFrame(data = list(range(0,self.num_to_generate)),
                     columns = [self.static_dataset.processor.mappings.index])
        df = pd.concat([id_column,df],axis = 1)
        df.columns = self.static_dataset.processed_df_columns

        processor = self.static_dataset.get_processor()
        df = processor.reverse(df)

        return df


def restore_manually(static_dataset, generated_raw_samples) :
    df = pd.DataFrame(generated_raw_samples)
    id_column = pd.DataFrame(data = list(range(0, len(generated_raw_samples))),
                             columns = [static_dataset.processor.mappings.index])
    df = pd.concat([id_column, df], axis = 1)
    df.columns = static_dataset.processed_df_columns

    processor = static_dataset.get_processor()
    df = processor.reverse(df)
    return df


        
        
#%%




        
        

        
        



        

        
