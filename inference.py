'''
Copy this code and save it in a file named inference.py, and place that file in SeqDeepFake folder.
'''

from models.configuration import Config
from models import SeqFakeFormer
from models.configuration import Config

from types import SimpleNamespace
import torch
import json
from PIL import Image
import torchvision.transforms as transforms

# Define label dictionaries
component_labels = {
    0: 'NA',
    1: 'nose',
    2: 'eye',
    3: 'eyebrow',
    4: 'lip',
    5: 'hair'
}

attribute_labels = {
    0: 'NA',
    1: 'Bangs',
    2: 'Eyeglasses',
    3: 'Beard',
    4: 'Smiling',
    5: 'Young'
}

def create_caption_and_mask(cfg):
    caption_template = cfg.PAD_token_id*torch.ones((1, cfg.max_position_embeddings), dtype=torch.long).cuda()
    mask_template = torch.ones((1, cfg.max_position_embeddings), dtype=torch.bool).cuda()

    caption_template[:, 0] = cfg.SOS_token_id
    mask_template[:, 0] = False

    return caption_template, mask_template

def get_inference(cfg, model_path, img):
  model = SeqFakeFormer.build_model(cfg)
  state_dictionary = torch.load(model_path, map_location='cpu')
  checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
  model.load_state_dict(checkpoint['best_state_dict_fixed'])

  # Load the image
  image = Image.open(img).convert('RGB')

  # Define transformations
  transform = transforms.Compose([
      transforms.Resize((256, 256)),
      transforms.ToTensor(),
  ])

  img_tensor = transform(image)
  img_tensor = img_tensor.to(torch.device("cuda"))
  if not isinstance(img_tensor, list):
    img_tensor = [img_tensor]

  caption, cap_mask = create_caption_and_mask(cfg)
  caption = caption.to(torch.device("cuda"))
  cap_mask = cap_mask.to(torch.device("cuda"))

  model = model.to(torch.device("cuda"))
  model.eval()
  with torch.no_grad():
    for i in range(cfg.max_position_embeddings - 1):
      predictions = model(img_tensor, caption, cap_mask)
      predictions = predictions[:, i, :]
      predicted_id = torch.argmax(predictions, axis=-1)

      if predicted_id[0] == cfg.EOS_token_id:
          caption = caption[:, 1:]
          zero = torch.zeros_like(caption)
          caption = torch.where(caption==cfg.PAD_token_id, zero, caption)
          break

      caption[:, i+1] = predicted_id[0]
      cap_mask[:, i+1] = False

    if caption.shape[1] == 6:
        caption = caption[:, 1:]
  return caption


def main():
  model_path = "" #Enter the model weights
  img_path = "" #Test Image Path
  cfg_path = "" #Config file path

  cfg = Config(cfg_path)

  tensor_result = get_inference(cfg, model_path, img_path)

  if tensor_result.dim() == 2:
    tensor_result = tensor_result.squeeze().tolist()
  else:
      tensor_result = tensor_result.tolist()

if __name__ == "__main__":
  main()
