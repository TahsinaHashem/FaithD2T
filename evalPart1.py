cache_dir = input("Please enter the location of cache dir:")

#Laoding saved PEFT Model.....

from transformers import Blip2ForConditionalGeneration, AutoProcessor
from peft import PeftModel, PeftConfig

import requests
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
# #
peft_model_id = "Part1-blip2-saved-modelFlanT5-XL" # #saved trained model
config = PeftConfig.from_pretrained(peft_model_id)

#for Flan-T5-XL load the model in load in 8bit format
model = Blip2ForConditionalGeneration.from_pretrained(config.base_model_name_or_path, load_in_8bit=True, cache_dir=cache_dir, device_map="auto") #,
processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl", cache_dir=cache_dir) 

######################################################################
# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id, cache_dir=cache_dir, device_map={"":0} ) #device_map="auto"
model.eval()

print("Peft model loaded")


##################################################################################################
# #######################################Reading Image links of Every sample Total 53219 samples##############################
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


fileFeatures = open("Second100_GraphFeaturesFiltered_MiniGPT4Summary.txt","r",encoding='ISO-8859-1')

LinesF = fileFeatures.readlines()
Feature=[]

for line in LinesF:
    if line != "\n":
          # print("Line from Reading FIle:",line)
          if line.startswith("Sample"):
              # print(line)
              listFeature = []
              # isFirstLine=True
          elif line.startswith("END"): ##END
              Feature.append(listFeature)
          else:
              listFeature.append(line.strip())

print("Total Sample Features: "+str(len(Feature)))



for id in range(0,len(Feature)):
    print("Sample:"+str(id))
    print( Feature[id])


saveResultsFile = "Second100_Mapping_GraphFeaturesFiltered_MiniGPT4.txt"

filePart1Result = open(saveResultsFile,"a")


for id in range(0,len(Feature)):
    listF = Feature[id]
    img=GraphImageArr[id+43100]
    print("Sample:"+str(id+43100))
    filePart1Result.write("Sample:")
    filePart1Result.write(str(id+43100))
    filePart1Result.write("\n")

    for eachF in listF:
        filePart1Result.write(str(eachF))
        filePart1Result.write("::")
        ans=[]
        for eachImg in img:
            ques="Is the feature -'" + str(eachF) + "' 'Salient' or 'Not Salient' or 'Hallucinated'?"

            question = "Question:" + ques + " Short answer:"
            

            image_path = eachImg.strip()

            image = Image.open(requests.get(image_path, stream=True).raw).convert('RGB')
            #
            inputs = processor(image, question, return_tensors="pt").to(device)
            #
            out = model.generate(**inputs, max_new_tokens=50)
            #   # print("Original Caption:" + text)
            generated_text = processor.decode(out[0], skip_special_tokens=True)
            #
            print(str(question))
            # print("Original Caption:" + text)
            print(str(generated_text))

            ans.append(generated_text)

            if generated_text.startswith("Salient") or generated_text.startswith("'Salient'"):
                break

        finalRes=""
        #Saliency Check
        anySalient=False
        for a in ans:
            a=a.strip()
            if a.startswith("Salient") or a.startswith("'Salient'"):
                anySalient=True
                finalRes="Salient"
                break

        # Not Saliency Check
        if anySalient == False:
            anyNotSalient=False
            for a in ans:
                a = a.strip()
                if a.startswith("Not Salient") or a.startswith("'Not Salient'"):
                    anyNotSalient = True
                    finalRes = "Not Salient"
                    break

            if anyNotSalient == False:
                finalRes = "Hallucinated"


        print("Final Result:"+str(finalRes))
        
        filePart1Result.write(str(finalRes))
        filePart1Result.write("\n")

    filePart1Result.write("END\n")
#
filePart1Result.close()
