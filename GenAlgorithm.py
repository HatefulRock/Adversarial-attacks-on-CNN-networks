import cv2 
import matplotlib.pyplot as plt 
import numpy as np

from test_LPRNet import *


image = cv2.imread('沪AMS087.jpg')
path="./data/GeneratedImages"
#cv2.imwrite(os.path.join(path , 'waka.jpg'), image)


colors={"white":[165,190,241],"blue":[40,99,246]}
n_iter=10
pixelsmodified=0
totalpixels=94*24


#show image
#cv2.imshow("before",image)


# cv2.imshow("after",image)
# cv2.waitKey()

#create 10 new images that all have one pixel randomly modified
def createNewImages(img):
    global pixelsmodified
    newImage = img.copy()
    mask = np.zeros((img.shape[0],img.shape[1],img.shape[2]),dtype=np.uint8)
    x = np.random.randint(0,img.shape[0])
    y = np.random.randint(0,img.shape[1])
    #modify color of new image with random color from colors
    mask[x,y]=colors[np.random.choice(list(colors.keys()))]
    newImage = cv2.add(newImage,mask)
    pixelsmodified+=1
    return [newImage,mask]

def numberOfPixelsModified(mask):
    #count the number of pixels that are different than 0
    return np.count_nonzero(mask)



def Fitness(lprnet):
    fitness = []
    lab=[]
    lab=test(lprnet)
    lab=np.array(lab)
    
    for i in range(len(lab)):
        fitness.append(min(np.subtract(lab[i,0,:,1],lab[i,1,:,1]))/(1-pixelsmodified/totalpixels))#probleme il faudrit multiplier par le nombre de pixels pour avoir un fitness qui tend vers le min

    return fitness


def main():
    global pixelsmodified
    args = get_parser()
    lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)
    device = torch.device("cpu")
    lprnet.to(device)
    print("Successful to build network!")

    # load pretrained model
    if args.pretrained_model:
        lprnet.load_state_dict(torch.load(args.pretrained_model,map_location=torch.device('cpu')))
        print("load pretrained model successful!")
    else:
        print("[Error] Can't found pretrained mode, please check!")
        return False
    
    population=[createNewImages(image) for i in range(10)]#creates a list of all the new images and their masks
    #show the first image in the population and its mask

    print("wait")
    #print(population)
    for gen in range(n_iter):
        #calculate fitness of each image
        #add images to folder GeneratedImages
        for i,v in enumerate(population):
            cv2.imwrite(path+"/沪AMS087"+"_img"+str(i)+".jpg",v[0])
        print("Generation number:",gen)
        print("Pixels modified:",pixelsmodified)
        fitnesses = np.array(Fitness(lprnet))
        print("fitness",fitnesses)
        #erase all images in folder GeneratedImages
        for i in range(10):
            os.remove(path+"/沪AMS087"+"_img"+str(i)+".jpg")
        #select the top 5 images
        population = [population[i] for i in fitnesses.argsort()[:5]]
        #crossover
        for i in range(5):
            for j in range(5):
                #create new image by adding the masks of the two parents
                newImage = population[i][0]+population[j][1]
                population.append([newImage,population[i][1]+population[j][1]])
        for i,v in enumerate(population):
            pixelsmodified+=numberOfPixelsModified(population[i][1])
        #mutation
        for i in range(len(population)):
                #replace population with mutated population images with createNewImages
                population[i] = createNewImages(population[i][0])
    #get images with highest fitness
    for i,v in enumerate(population):
            cv2.imwrite(path+"/沪AMS087"+"_img"+str(i)+".jpg",v[0])
    fitnesses = np.array(Fitness(lprnet))
    bestImage = population[fitnesses.argsort()[-1]]

    #show best image
    cv2.imshow("after",bestImage[0])
    cv2.imshow("mask",bestImage[1])
    cv2.waitKey()


if __name__ == "__main__":
    main()


    
