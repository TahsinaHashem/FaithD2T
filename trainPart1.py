cache_dir = input("Please enter the location of cache dir:")

from datasets import load_dataset 

dataset_location=input("Please enter the location of training dataset:")

train_dataset = load_dataset("json", data_files=dataset_location, split="train[:88%]") #split="train[:88%]
val_dataset = load_dataset("json", data_files=dataset_location, split="train[:88%]") #split="train[:88%]

print("Training sets: {} - Validating set: {}".format(len(train_dataset), len(val_dataset)))

import requests
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
# # # # #
class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        imgPath = item["imgPath"]
        
        raw_image = Image.open(requests.get(imgPath, stream=True).raw).convert('RGB')
        
        question = "Question:" + item["question"] + " Short answer:"
        
        encoding = self.processor(images=raw_image, text=question, padding="max_length", max_length=100, return_tensors="pt")
        # remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        
        encoding["labels"] = item["answer"] + ". Explanation: "+ item["explanation"]
        return encoding

def collate_fn(batch):
    # pad the input_ids and attention_mask
    processed_batch = {}
    for key in batch[0].keys():
        if key != "labels":
            processed_batch[key] = torch.stack([example[key] for example in batch])
        else:
            text_inputs = processor.tokenizer(
                [example["labels"] for example in batch], padding=True, return_tensors="pt"
            )
            processed_batch["labels"] = text_inputs["input_ids"]
            processed_batch["attention_mask"] = text_inputs["attention_mask"]
    return processed_batch
#
# # #
from transformers import AutoProcessor, Blip2ForConditionalGeneration
#
#
processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl", cache_dir = cache_dir)
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl",cache_dir = cache_dir, load_in_8bit=True, device_map="auto")


# # ###################### Finetuning BLIP2 Flan-t5 Encoder-Decoder Model PEFT based #########################################################
# # #
from peft import LoraConfig, get_peft_model,TaskType, prepare_model_for_int8_training
#
# # Define LoRA Config
lora_config = LoraConfig(
 r=16,
 lora_alpha=32,
 target_modules=["q", "v"],
 lora_dropout=0.05,
 bias="none",
 # task_type=TaskType.SEQ_2_SEQ_LM
)
#
 # prepare int-8 model for training
model = prepare_model_for_int8_training(model)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
#
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

train_dataset = ImageCaptioningDataset(train_dataset, processor)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16, collate_fn=collate_fn)

valid_dataset = ImageCaptioningDataset(val_dataset, processor)
valid_dataloader = DataLoader(valid_dataset, shuffle=True, batch_size=2, collate_fn=collate_fn)
# #
import torch
from tqdm import tqdm
# #
#
# #
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

model.train()
min_eval_loss = float("inf") ##added
patience = 10
early_stopping_hook = 0
#
for epoch in range(100):
    print("Epoch:", (epoch+1))
    epoch_loss = 0
    for idx, batch in zip(tqdm(range(len(train_dataloader)), desc='Training batch: ...'), train_dataloader):
        input_ids = batch.pop("input_ids").to(device)
        labels = batch.pop("labels").to(device)
        pixel_values = batch.pop("pixel_values").to(device)

        outputs = model(input_ids=input_ids,
                    pixel_values=pixel_values,
                    labels=labels)

        loss = outputs.loss
        epoch_loss += loss.item()

        # print("Loss:", loss.item())

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    eval_loss = 0
    for idx, batch in zip(tqdm(range(len(valid_dataloader)), desc='Validating batch: ...'), valid_dataloader):
            input_ids = batch.pop('input_ids').to(device)
            pixel_values = batch.pop('pixel_values').to(device)
            attention_masked = batch.pop('attention_mask').to(device)
            labels = batch.pop('labels').to(device)

            # with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(input_ids=input_ids,
                                pixel_values=pixel_values,
                                #attention_mask=attention_masked,
                                labels=labels)

            loss = outputs.loss
            eval_loss += loss.item()
            # print("Validation Loss:", loss.item())

    # tracking_information.append((epoch_loss / len(train_dataloader), eval_loss / len(valid_dataloader), optimizer.param_groups[0]["lr"]))
    print("Epoch: {} - Training loss: {} - Eval Loss: {} - LR: {}".format(epoch + 1, epoch_loss / len(train_dataloader),
                                                                              eval_loss / len(valid_dataloader),
                                                                              optimizer.param_groups[0]["lr"]))
        # scheduler.step()
    if eval_loss < min_eval_loss:
        model.save_pretrained("Part1-blip2-saved-modelFlanT5-XL", from_pt=True)
        #print("Saved model to Part1-blip2-saved-modelFlanT5-XL")
        min_eval_loss = eval_loss
        early_stopping_hook = 0
    else:
        early_stopping_hook += 1
        if early_stopping_hook > patience:
            break

    model.train()
#
print("TRaining Done....Enter to continue.....")
#

