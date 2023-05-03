# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 17:33:33 2022

@author: brend
"""

import aicspylibczi as czi
import numpy as np
import math
import scipy.signal
import tifffile as tif
import PySimpleGUI as sg
import os.path
import statistics 
import time
import matplotlib.pyplot as plt
import csv

def makeGUI():

    file_list_column = [
        [
            sg.Text("Image Folder"),
            sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
            sg.FolderBrowse(),
            ],
        [
            sg.Listbox(
                values=[], select_mode=sg.LISTBOX_SELECT_MODE_MULTIPLE, enable_events=True, size=(40, 20), key="-FILE LIST-"
                )
            ],
        ]

    image_viewer_column = [
        [sg.Text("Choose an image from list on left:")],
        [sg.Image(key="-IMAGE-")],
        [sg.Text('Laminin Channel Number (integer, index from 0)', size =(35, 1)), sg.InputText(key = "channelSelect", size = (8,1))],
        [sg.Text('Intensity Threshold (integer)', size =(21, 1)), sg.InputText(key = "threshold", size = (8,1))],
        [sg.Text('Tile Size (integer pixels)', size =(21, 1)), sg.InputText(key = "TileSize", size = (8,1))],
        [sg.Text('Tile Size (integer microns)', size =(21, 1)), sg.InputText(key = "TileSizeMicrons", size = (8,1))],
        [sg.Text('Pixel Pitch (um/pixel)', size =(21, 1)), sg.InputText(key = "Pitch", size = (8,1))],
        [sg.Text('Line Profile Number (integer)', size =(21, 1)), sg.InputText(key = "Profiles", size = (8,1))],
        [sg.Checkbox('Smoothing Enabled', default=False, key="Smooth")],
        [sg.Checkbox('Output Heatmap CSV', default=False, key="CSV")],
        [sg.Text('Output filepath', size =(16, 1)), sg.InputText(key = "path")],
        [sg.Button("Generate")]
        ]

    layout = [
        [
            sg.Column(file_list_column),
            sg.VSeperator(),
            sg.Column(image_viewer_column),
            ]
        ]

    window = sg.Window("Fibrotic Map Generator", layout)

    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if event == "-FOLDER-":
            folder = values["-FOLDER-"]
            try:
                file_list = os.listdir(folder)
            except:
                file_list = []
            
            fnames = [
                f
                for f in file_list
                if os.path.isfile(os.path.join(folder, f))
                and f.lower().endswith(('.tif', '.czi'))
                ]
            
            window["-FILE LIST-"].update(fnames)
        
        
        if event == "-FILE LIST-":  
            try:
                files = values["-FILE LIST-"]
            except:
                pass
        
        if event == "Generate":
            for filename in files:
                fullpath = os.path.join(values["-FOLDER-"], filename)
                FWHM_fibProfiler(fullpath, values["Profiles"], values["TileSize"], values['channelSelect'], values['path'], values["threshold"], values['Pitch'], values['TileSizeMicrons'], values["Smooth"], values["CSV"])

def Slicer(image, tileSize, channelSelect):
   
    #image loading and dimensional manipulation, selection of a single channel
    
    if image[-3:] == 'czi':
        file  = czi.CziFile(image)
        fileShape = file.get_dims_shape()
        channels = fileShape[0]['C']
        numberOfChannels = channels[1]-channels[0]
        img = file.read_mosaic(C=channelSelect)
        img=img.squeeze()
        pxHeight, pxWidth = np.shape(img)
        
    else:
        file = tif.imread(image)
        img = tif.imread(image);
        with tif.TiffFile(image) as tiff:
            imagej_metadata = tiff.imagej_metadata
            tags = tiff.pages[0].tags
            x_spacing = tags["XResolution"].value[0], tags["XResolution"].value[1]
            y_spacing = tags["YResolution"].value[0], tags["YResolution"].value[1]
        
        if len(np.shape(img)) == 2:
            numberOfChannels=1
            pxHeight, pxWidth = np.shape(img)
            file=np.expand_dims(file, axis=0)
        else:
            numberOfChannels, pxHeight, pxWidth = np.shape(img)
            img = img[channelSelect]
            img = img.squeeze()
        
    tiles = []
    tileCorner = []
    
    cnt=0
    for i in range(0, pxHeight-tileSize, tileSize):
        for j in range(0, pxWidth-tileSize, tileSize):
            tiles.append(img[i:i+tileSize, j:j+tileSize])
            tileCorner.append((i, j))
            cnt=cnt+1
            
    return tiles, tileSize, pxWidth, pxHeight, img, file, numberOfChannels, imagej_metadata, x_spacing, y_spacing, tileCorner

def FWHM_fibProfiler(image, lines, tileSize, channelSelect, path, threshold, pitch, microns, smoothing, CSV):
    st = time.time()
    start = image.index('\\')
    imageName = image[start+1:-4]

    if tileSize:
        tileSize = int(tileSize)
        if microns:
            micronSize = int(microns)*(1/float(pitch))
            if abs(micronSize - tileSize) >= tileSize*0.1:
                sg.popup('WARNING: Micron-Pixel Parameters Do Not Match')    
    elif microns:
        tileSize = int(microns)*(1/float(pitch))
        
    tileSize = math.floor(tileSize)
    
    lines = int(lines)
    
    channelSelect = int(channelSelect)
    threshold = int(threshold)
    
    tiles, tileSize, pxWidth, pxHeight, img, file, numberOfChannels, imagej_metadata, x_spacing, y_spacing, tileCorner = Slicer(image, tileSize, channelSelect)
    
    #first progress bar
    layout = [[sg.Text('Creating Line Profiles in X and Y Dimensions')],
             [sg.ProgressBar(len(tiles), orientation='h', size=(20, 20), key='progressbar')]]
    window = sg.Window('Progress Bar', layout)
    progress_bar = window['progressbar']
    #loops through all tiles and all line profiles in each tile, 
    #performs cubic spline interpolation to smooth data, 
    #finds peaks and widths of curve under peak     
    xValues = (np.linspace(0, tileSize-1, tileSize))
    xValues = [math.ceil(i) for i in xValues]
    def profileMaker(l):
        b, a = scipy.signal.ellip(4, 0.01, 120, 0.125)
        FWHM_ave = []
        if l == 'y':
            progress_bar(0)
        for k in range(0,len(tiles)):
            event, values = window.read(timeout=10)
            if event == 'Cancel'  or event == sg.WIN_CLOSED:
                break
            progress_bar.UpdateBar(k + 1)
            #data acquisition/clean up
            # print('At tile',k,'/',len(tiles))
            for m in range(1,lines):
                profiles = math.floor((tileSize/lines)*m)
                img = np.array(tiles[k])
                if l=='x':
                    lineProfile = img[:,profiles]
                if l=='y':
                    lineProfile = img[profiles,:]
                newData = scipy.signal.filtfilt(b, a, lineProfile, method = 'gust')
                crossedUp = []
                crossedDown = []
                widths = []
                
        
                for i in range(1, len(newData)):
                    if (newData[i-1] < threshold and newData[i] > threshold):
                        crossedUp.append(i)
                    if (newData[i-1] > threshold and newData[i] < threshold):
                        crossedDown.append(i)
                if len(crossedUp) == 0 and len(crossedDown) == 0:       
                    if all(newData < threshold):
                        FWHM_ave.append(0)
                    if all(newData > threshold):
                        FWHM_ave.append(int(tileSize))
                elif len(crossedUp) == 0 or len(crossedDown) == 0:
                    if len(crossedUp) == 0:
                        widths.append(crossedDown[0])
                    else:
                        widths.append(tileSize-crossedUp[-1])
                    FWHM_ave.append(sum(widths)/len(widths))
                elif len(crossedUp) == 1 and len(crossedDown) == 1:
                    if crossedDown[0] < crossedUp[0]:
                        widths.append(crossedDown[0])
                        widths.append(tileSize - crossedUp[0])
                    else:
                        widths.append(abs(crossedDown[0] - crossedUp[0]))
                    FWHM_ave.append(sum(widths)/len(widths))
                else:
                    if crossedDown[0] < crossedUp[0]:
                        widths.append(crossedDown[0])
                        if crossedUp[-1] < crossedDown[-1]:
                            for i in range(0, len(crossedUp)):
                                widths.append(abs(crossedDown[i+1]-crossedUp[i]))
                        if crossedUp[-1] > crossedDown[-1]:
                            widths.append(tileSize - crossedUp[-1])
                            for i in range(1, len(crossedUp)):
                                widths.append(abs(crossedDown[i] - crossedUp[i-1]))
                    if crossedDown[0] > crossedUp[0]:
                        if crossedUp[-1] > crossedDown[-1]:
                            widths.append(tileSize - crossedUp[-1])
                            for i in range(0, len(crossedDown)):
                                widths.append(abs(crossedDown[i] - crossedUp[i]))
                        if crossedUp[-1] < crossedDown[-1]:
                            for i in range(0, len(crossedUp)):
                                widths.append(abs(crossedDown[i] - crossedUp[i]))
                    FWHM_ave.append((sum(widths)/len(widths)))
        return FWHM_ave
            
    FWHM_aveX = profileMaker('x')
    FWHM_aveY = profileMaker('y')
    
    window.close()
    
    #takes each average from line profiles and groups them together 
    FWHMPerTileX = [FWHM_aveX[x:x+(lines-1)] for x in range(0, len(FWHM_aveX), (lines-1))] 
    FWHMPerTileY = [FWHM_aveY[x:x+(lines-1)] for x in range(0, len(FWHM_aveY), (lines-1))] 
    #averages each grouping to create an average for the tile
    avePerTileX = [(sum(each))/(len(each)) for each in FWHMPerTileX]
    avePerTileY = [(sum(each))/(len(each)) for each in FWHMPerTileY]
    
    avePerTile = []
    for each in range(0, len(avePerTileX)):
        avePerTile.append((avePerTileX[each]+avePerTileY[each])/2)
        
    for i in range(1, len(avePerTile)-1):
        avePerTile[i] = (avePerTile[i-1] + avePerTile[i] + avePerTile[i+1])/3
    
    tilesPerRow = math.floor((pxWidth-tileSize)/(tileSize))
    
    if smoothing == True:
        for i in range(1, len(avePerTile)-1):
            avePerTile[i] = (avePerTile[i-1] + avePerTile[i] + avePerTile[i+1])/3
        for i in range(tilesPerRow*2, len(avePerTile) - tilesPerRow*2):
            avePerTile[i] = ((avePerTile[i+tilesPerRow] + avePerTile[i] + avePerTile[i - tilesPerRow])/3)
    
    layout = [[sg.Text('Creating heat map')],
              [sg.ProgressBar(len(tiles), orientation='h', size=(20, 20), key='progressbar')],]
    window = sg.Window('Progress Bar', layout)
    progress_bar = window['progressbar']
    
    #assigning score to each tile
    for i in range(0, len(tiles)):
        event, values = window.read(timeout=10)
        if event == 'Cancel'  or event == sg.WIN_CLOSED:
                break
        progress_bar.UpdateBar(i + 1)
        
        tiles[i][:][:] = avePerTile[i]
    
    window.close()
    
    layout = [[sg.Text('Saving new image')],
              [sg.ProgressBar(len(tiles), orientation='h', size=(20, 20), key='progressbar')]]
    window = sg.Window('Progress Bar', layout)
    progress_bar = window['progressbar']
    
    #re-creating image with same dimensions as input 
    newImage = np.zeros((pxHeight, pxWidth), dtype = "uint16") 
    cnt = 0
    for i in range(0, pxHeight-tileSize, tileSize):
        for j in range(0, pxWidth-tileSize, tileSize):
            event, values = window.read(timeout=10)
            if event == 'Cancel'  or event == sg.WIN_CLOSED:
                break
            progress_bar.UpdateBar(i + 1)
            newImage[i:i+tileSize, j:j+tileSize] = tiles[cnt][:][:]
            cnt = cnt+1
            
    window.close()
    
    info = {"Tile Size" : tileSize, 
            "Threshold": threshold, 
            "Line Profile Number": lines, 
            "Laminin Channel Number": channelSelect, 
            "Smoothing Enabled": smoothing, 
            "Output Path": path}
    
    if CSV:
        rows = zip(tileCorner, avePerTile)
        with open(os.path.join(path, imageName+"TileMap.csv"), "w") as f:
            writer = csv.writer(f)
            w = csv.DictWriter(f, info.keys())
            w.writeheader()
            w.writerow(info)
            for row in rows:
                writer.writerow(row)
            
        
    #exporting image as an additional channel on original image as tiff
    newImage = np.expand_dims(newImage, axis = 0)
    finalImg = np.concatenate((newImage, file)) 
    tif.imwrite(os.path.join(path, imageName+"_Scored.tif"), finalImg, imagej=True, metadata = imagej_metadata, resolution=[x_spacing, y_spacing])
    et = time.time()
    time_elapsed = et-st
    print(time_elapsed)
    
makeGUI()