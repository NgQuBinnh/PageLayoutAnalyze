import json

data_set = open('D:\Coding\\tensorflow-deeplab-resnet\dataset\\test.txt', 'r').readlines()
images = [image.split(' ')[0].replace("/JpegImages/", "").replace(".jpg", "") for image in data_set]

with open('predicted_boxes.json') as json_data:
    d = json.load(json_data)
    save_dir = 'D:\Coding\Object-Detection-Metrics\detections'

    for idx, image in enumerate(images):
        if idx == 38:
            break
        ws = open(image + ".txt", "w")
        list_labels = d[image]['labels']
        list_bbxs = d[image]['boxes']
        list_conf = d[image]['confidence_score']
        for idx,label in enumerate(list_labels):
            bbx = list_bbxs[idx]
            ws.write(list_labels[idx] + " " + str(list_conf[idx]) + " " +
                     str(bbx[0]) + " " + str(bbx[1]) + " " +
                     str(bbx[2]) + " " + str(bbx[3]) + "\n")
        ws.close()

