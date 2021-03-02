import pydicom as dicom, numpy as np, multiprocessing as mp, matplotlib.pyplot as plt
import os, cv2, sys, time
from pydicom.pixel_data_handlers.util import convert_color_space, apply_color_lut
from pydicom.errors import InvalidDicomError

t_mark_template_path = "T_mark_template.png"
t_mark_template = cv2.imread(t_mark_template_path)

def getFilelist(input_folder):
    filelist = []

    for filename in os.listdir(input_folder):
        if os.path.isdir(os.path.join(input_folder, filename)):
            templist = getFilelist(os.path.join(input_folder, filename))
            for temp in templist:
                filelist.append(temp)
        else:
            if filename == "DICOMDIR" or (len(filename.split(".")) >= 2 and not filename.split(".")[-1] == "dcm"):
                continue
            filelist.append(os.path.join(input_folder, filename))
    return filelist

def makeOutputFolders(input_folder, output_folder, filelist):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for filepath in filelist:
        internal_folders = filepath[len(input_folder) : filepath.rfind("\\")]
        temp = output_folder
        for folder in internal_folders.split("\\"):
            temp = os.path.join(temp, folder)
            if not os.path.exists(temp):
                os.mkdir(temp)

def getOutputlist(input_folder, output_folder, filelist):
    outputlist = []
    for filepath in filelist:
        outputlist.append(output_folder + "/" + filepath[len(input_folder) : filepath.rfind("\\")])
    return outputlist

def find_template(img):
    w = t_mark_template.shape[1]
    h = t_mark_template.shape[0]
    # Apply template Matching
    method = cv2.TM_CCOEFF
    res = cv2.matchTemplate(img, t_mark_template, method)
    _, _, min_loc, max_loc = cv2.minMaxLoc(res)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    return top_left, bottom_right

def removeWhiteNoises_frame(frame):
    if frame.dtype == 'uint16':
        frame = np.uint8(frame)
    temp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Remove white noises around edge
    retval, labels = cv2.connectedComponents(temp)
    blockSet = []
    for i in range(retval):
        blockSet.append([])
        
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if not labels[i][j] == 0:
                blockSet[labels[i][j]].append(i*labels.shape[1]+j)
    
    noises_pointer_list = []
    for block in blockSet:
        x = frame.shape[1]
        y = frame.shape[0]
        width = 0
        height = 0
        for value in block:
            tempX = int(value%frame.shape[1])
            tempY = int(value/frame.shape[1])
            x = min(x, tempX)
            y = min(y, tempY)
            width = max(width, tempX)
            height = max(height, tempY)
        
        if (x <= 2 or y <= 2) and (width < frame.shape[1]/100*5 or height < frame.shape[0]/100*5):
            noises_pointer_list.append(block)
    for block in noises_pointer_list:
        for value in block:
            tempX = int(value%frame.shape[1])
            tempY = int(value/frame.shape[1])
            frame[tempY, tempX] = 0

    # Remove the T mark
    top_left, bottom_right = find_template(frame)
    frame[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0], :] = np.zeros((bottom_right[1] - top_left[1], bottom_right[0] - top_left[0], 3))

    return frame

def removeWhiteNoisesAndSave(framelist, filename, focus_depth, index, outputFolder):
    if framelist.ndim == 4: # if the data is video
        for i in range(framelist.shape[0]):
            frame = framelist[i]
            frame = removeWhiteNoises_frame(frame)
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            outputpath = os.path.join(outputFolder, filename.split(".")[0] + "_" + str(focus_depth) + "_Frame" + str(i+index) + ".png")
            cv2.imwrite(outputpath, frame)
    else:
        frame = framelist
        frame = removeWhiteNoises_frame(frame)
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        outputpath = os.path.join(outputFolder, filename.split(".")[0] + "_" + str(focus_depth) + "_Frame" + str(index) + ".png")
        cv2.imwrite(outputpath, frame)

def process(filelist, outputlist, isMultiprocessing=True, p_no=1, isOnlyUSImage=False):
    print("isMultiprocessing:", isMultiprocessing)
    for serial_no, filepath in enumerate(filelist):
        try:
            ds = dicom.dcmread(filepath)
        except InvalidDicomError as e:
            print("Error:", e)
        # try:
        #     ds[0x00081080].value
        # except KeyError as e:
        #     print("Error:", e)
        #     continue
        
        try:
            pixel_array = ds.pixel_array
        except Exception:
            with open("Error.txt", "a") as f:
                f.write(filepath) 
            continue
        print(filepath)
        photometricInterpretation = ds[0x00280004].value
        if photometricInterpretation == "YBR_FULL_422":
            pixel_array = convert_color_space(pixel_array, current="YBR_FULL_422", desired="RGB")
        elif photometricInterpretation == "YBR_FULL":
            pixel_array = convert_color_space(pixel_array, current="YBR_FULL", desired="RGB")
        elif photometricInterpretation == "PALETTE COLOR":
            img = []
            for i in range(pixel_array.shape[0]):
                img.append(apply_color_lut(pixel_array[i], ds))
            pixel_array = np.array(img)

        if pixel_array.ndim == 4: # if the data is video
            # Get some information
            print(pixel_array.shape)
            filename = filepath.split("\\")[-1]
            focus_depth = -1
            if 0x00185012 in ds:
                focus_depth = ds[0x00185012].value
            if isOnlyUSImage:
                us_sequence = ds[0x00186011][0]
                x0_region = us_sequence[0x00186018].value
                y0_region = us_sequence[0x0018601a].value
                x1_region = us_sequence[0x0018601c].value
                y1_region = us_sequence[0x0018601e].value
                
                pixel_array = pixel_array[:, y0_region : y1_region, x0_region : x1_region, :]
                # x0_offset = 35
                # x1_offset = 795
                # pixel_array = pixel_array[:,  : , x0_offset : x1_offset, :]

                if isMultiprocessing:
                    p_list = []
                    for i in range(p_no):
                        p_list.append(mp.Process(target=removeWhiteNoisesAndSave, 
                        args=(pixel_array[int(i*len(pixel_array)/p_no) : int((i+1)*len(pixel_array)/p_no)], 
                        filename, focus_depth, int(i*len(pixel_array)/p_no), outputlist[serial_no])))
                    for i in range(p_no):
                        p_list[i].start()
                    for i in range(p_no):
                        p_list[i].join()
                else:
                    removeWhiteNoisesAndSave(pixel_array, filename, focus_depth, 0, outputlist[serial_no])
            else:
                for i in range(pixel_array.shape[0]):
                    outputpath = os.path.join(outputlist[serial_no], 
                        filename.split(".")[0] + "_" + str(focus_depth) + "_Frame" + str(i) + ".png")
                    cv2.imwrite(outputpath, cv2.cvtColor(pixel_array[i], cv2.COLOR_RGB2BGR))
                
        elif pixel_array.ndim == 3: # if the data is picture
            print(pixel_array.shape)
            filename = filepath.split("\\")[-1]
            focus_depth = -1
            if 0x00185012 in ds:
                focus_depth = ds[0x00185012].value 
            if isOnlyUSImage:
                us_sequence = ds[0x00186011][0]
                x0_region = us_sequence[0x00186018].value
                y0_region = us_sequence[0x0018601a].value
                x1_region = us_sequence[0x0018601c].value
                y1_region = us_sequence[0x0018601e].value
                
                pixel_array = pixel_array[:, y0_region : y1_region, x0_region : x1_region, :]
                # x0_offset = 35
                # x1_offset = 795
                # pixel_array = pixel_array[:, x0_offset : x1_offset, :]
                removeWhiteNoisesAndSave(pixel_array, filename, focus_depth, 0, outputlist[serial_no])
            else:
                outputpath = os.path.join(outputlist[serial_no], 
                        filename.split(".")[0] + "_" + str(focus_depth) + "_Frame0.png")
                cv2.imwrite(outputpath, cv2.cvtColor(pixel_array, cv2.COLOR_RGB2BGR))
        elif pixel_array.ndim == 2: # if the data is picture and in grayscale
            print(pixel_array.shape)
            filename = filepath.split("\\")[-1]
            outputpath = os.path.join(outputlist[serial_no], 
                        filename.split(".")[0] + ".png")
            cv2.imwrite(outputpath, pixel_array)

    open("state.txt", "w").write("0")

def preprocess(input_folder, output_folder, isMultiprocessing, p_no, isOnlyUSImage):
    filelist = getFilelist(input_folder)
    makeOutputFolders(input_folder, output_folder, filelist)
    outputlist = getOutputlist(input_folder, output_folder, filelist)

    process(filelist, outputlist, isMultiprocessing, p_no, isOnlyUSImage)

if __name__ == "__main__":
    '''
    Put all dicom files in input folder and 
    will output all dicom files with video frame by frame to output folder.
    File name format : "input file name" + _ + "Focus depth" + "_" + "frame index" + .png
    '''
    # arguments = sys.argv[1:]
    # print(arguments)

    # input_folder = arguments[0]

    # output_folder = arguments[1]

    # isMultiprocessing = True
    # if arguments[2] == "False":
    #     isMultiprocessing = False
        
    # p_no = int(arguments[3])

    # isOnlyUSImage = False
    # if arguments[4] == "1":
    #     isOnlyUSImage = True
    
    # filelist = getFilelist(input_folder)
    # makeOutputFolders(input_folder, output_folder, filelist)
    # outputlist = getOutputlist(input_folder, output_folder, filelist)

    # process(filelist, outputlist, isMultiprocessing, p_no, isOnlyUSImage)


    original = "dicom"
    output_folder = "jpgs"
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    import time
    startTime = time.time()
    preprocess(os.path.join(original), os.path.join(output_folder), isMultiprocessing=True, p_no=4, isOnlyUSImage=False)
    print("Taking Time:", time.time() - startTime, "seconds.")