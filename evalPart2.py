#
cache_dir = input("Please enter the location of cache dir:")

import requests
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch

#Laoding saved PEFT Model.....
from transformers import Blip2ForConditionalGeneration, AutoProcessor
from peft import PeftModel, PeftConfig
#

peft_model_id = "Part2-blip2-saved-modelFlanT5-XL" #saved trained model
config = PeftConfig.from_pretrained(peft_model_id)
# # # # # # # #
model = Blip2ForConditionalGeneration.from_pretrained(config.base_model_name_or_path, load_in_8bit=True, cache_dir=cache_dir, device_map="auto") #,
# # #Load the LORA Model
model = PeftModel.from_pretrained(model, peft_model_id, cache_dir=cache_dir, device_map="auto")
model.eval()
print("Peft model loaded")

processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl", cache_dir=cache_dir) 
 ######################################Reading Image links of Every sample ##############################
GraphImageArr = []
fileImg = open('FullLinearGraphPicture.txt', 'r')
LinesImg = fileImg.readlines()

img=[]
index = -1
for limg in LinesImg:
    if limg.startswith("Sample") or limg.startswith("END"):
        if index != -1:
            GraphImageArr.append(img)
        img=[]
        index=index+1
      
        continue
    else:
        if limg.startswith("http://img.youtube.com"):
            continue
        
        if "mediaviewer_v3" in limg:
            continue
        img.append(limg)
        
#
print("Length of GraphImage Array:"+str(len(GraphImageArr)))


fileImg.close()

saveResultsFile = "SalientImgFeatures_TestSamples.txt"

filePart2Result = open(saveResultsFile,"a")


count=-1
for itemImg in GraphImageArr:
    count=count+1
    if count == 43200:
        break
    elif count >= 43100:
        filePart2Result.write("Sample:")
        filePart2Result.write(str(count))
        filePart2Result.write("\n")

        imagesArr = GraphImageArr[count]
        for imageLink in imagesArr:
            filePart2Result.write("\nImage Salient Features:\n")
            image = Image.open(requests.get(imageLink.strip(), stream=True).raw).convert('RGB')
            #for trained model
            question="Question: List the Salient Features of this Image for House Advertising ? Short answer:"
            #for demo Model
            #question = "Question: List Important Features of this Image for House Advertising ? Short answer:"

            inputs = processor(image, question, return_tensors="pt").to(device)
            #
            out = model.generate(**inputs,max_new_tokens=28)  

            generated_text = processor.decode(out[0], skip_special_tokens=True)
            #
            print(str(question))
            print("Output:" + str(generated_text))
            filePart2Result.write(str(generated_text))

        filePart2Result.write("\nEND\n")


filePart2Result.close()