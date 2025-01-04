import sys
import pandas as pd

from tld.data import main
from tld.configs import DataDownloadConfig

if __name__ == "__main__":
    data_link = "https://huggingface.co/datasets/zzliang/GRIT/resolve/main/grit-20m/coyo_0_snappy.parquet?download=true"

    data_config = DataDownloadConfig(
            data_link=data_link,
            latent_save_path="latent_folder",
            raw_imgs_save_path="raw_imgs_folder",
            download_data=True,
            number_sample_per_shard=50,
            batch_size=16,
            first_n_rows=100,
            use_wandb=False
            )

    main(data_config)


    from PIL import Image
    import torch
    from torchvision.transforms import Resize, ToTensor
    from datasets import load_dataset
    dataset_name = 'lambdalabs/pokemon-blip-captions'

    dataset = load_dataset(dataset_name)
    print(f"{dataset_name}: {dataset})

    def transform(example):
        example['image'] = example['image'].resize((256, 256), Image.BICUBIC)

        return example

    dataset["train"] = dataset["train"].map(transform)

    def custom_collate_fn(batch):
        images, texts = [], []
        for item in batch:
            image, text = item['image'], item['text']

            # Apply the transformation
            image = ToTensor()(image)

            images.append(image)
            texts.append(text)


        return torch.stack(images), texts

    #dataset.set_format(type='torch', columns=['image', 'text'])
    train_dataloader = torch.utils.data.DataLoader(dataset["train"], batch_size=32, shuffle=False, collate_fn=custom_collate_fn, num_workers=0)

    from tld.data import get_text_and_latent_embedings
    import clip
    from diffusers import AutoencoderKL
    import os
    import numpy as np

    latent_save_path = 'pokemon_path'

    model, preprocess = clip.load("ViT-L/14")

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    vae = vae.to('cuda')
    model.to('cuda')

    if not os.path.exists(latent_save_path):
        os.mkdir(latent_save_path)

    img_encodings, text_encodings, text_captions = get_text_and_latent_embedings(train_dataloader, vae, model, drive_save_path=latent_save_path)

    np.save(os.path.join(latent_save_path, 'image_latents.npy'), img_encodings.numpy())
    np.save(os.path.join(latent_save_path, 'text_encodings.npy'), text_encodings.numpy())

    creature_descriptions = [
        "A drawing of a small, blue aquatic creature with a fin on its head and a light blue tail.",
        "A picture of a fiery orange and red mythical dragon-like figure, with smoke billowing from its nostrils.",
        "A cartoon image of a character that looks like a yellow sunflower with a smiling face in the center.",
        "An illustration of a rock-like creature, gray and rugged, with crystals emerging from its back.",
        "A sketch of a ghostly figure, transparent and white, with glowing red eyes and ethereal trails.",
        "A drawing of a cute, furry, brown bear cub-like character, with large, round ears and a small nose.",
        "An image of an electric-type creature, bright yellow with black stripes, radiating energy.",
        "A picture of an ice-like character, resembling a small, crystalline snowflake with a shimmering, icy body."
    ]


    np.save('eval_encs.npy', encode_text(creature_descriptions, model))
