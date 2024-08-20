# -*- coding: utf-8 -*-
"""
the Functions for land use allocation, visualizing land use allocation, calculating objective's value

@author: Liu Muyang
"""
import  numpy as np
import math

import torch


# 此函数的作用是：初始化地块的用地类型，计算用地类型的相容性，计算用地类型的紧凑性，计算用地类型的内部可达性，计算用地类型的外
def map_to_strings(int_array, landtypes):

    mapping = {i: typ for i, typ in enumerate(landtypes)}
    string_list = []
    for i in int_array:
        string_list.append(mapping[i])
    return string_list
def map_to_num(strings, landtypes):

  mapping = {typ: i for i, typ in enumerate(landtypes)}
  ints = np.array([mapping[s] for s in strings])
  return ints

def get_adjacency_matrix(file, polygoncount):
    ''' Generate adjacency matrix from a CSV file '''
    import csv
    import numpy as np
    # 初始化邻接矩阵为全0，大小为 polygoncount x polygoncount
    adjacency_matrix = np.zeros((polygoncount, polygoncount), dtype=int)

    with open(file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')

        for row in spamreader:
            land1, land2 = [int(x.strip()) - 1 for x in row]
            # 更新邻接矩阵
            adjacency_matrix[land1][land2] = 1
            adjacency_matrix[land2][land1] = 1  # 因为邻接关系是双向的

    return adjacency_matrix
def getneighbourlist(file, polygoncount):
    ''' neighbour to list '''
    #the __file__ is the csv file in specific format including the information of all the neighbour relationships  
    #the __polygoncont__ is the number of polygons need to be allocated
    #finally, it will return a list of all the neighbour relationships between all the polygons
   
    import csv
    from copy import deepcopy
    with open(file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        center = 1
        neighbourdata = [[0 for x in range(1)] for y in range(polygoncount)]
        #创建一个二维数组，每个数组的第一个元素为该地块的邻居数，后面的元素为该地块的邻居地块的ID
        for row in spamreader: #循环读取csv文件中的每一行
             info = deepcopy(', '.join(row).split(',')) #将每一行的数据以逗号分隔开，存储在info中
             #后面的操作是将每一行的数据存储在neighbourdata中，neighbourdata的每一行代表一个地块，第一个元素为该地块的邻居数，后面的元素为该地块的邻居地块的ID
             if int(info[0]) == center:
                 neighbourdata[center-1].append(int(info[1])-1)
             else:
                 neighbourdata[center-1][0] = len(neighbourdata[center-1])-1
                 center += 1 
                 neighbourdata[center-1].append(int(info[1])-1)
    neighbourdata[-1][0] = len(neighbourdata[center-1])-1
    return neighbourdata


def readlanduselist(shapefile, polygoncount):
    ''' landuse shapefile to list '''
    #the __shapefile__ is the shapefile with land use allocation information, which will be applied in the programme 
    #the __polygoncont__ is the number of polygons need to be allocated
    #finally, it will return a list of land use allocation information as the shapefile included

    import geopandas as gpd
    import matplotlib.pyplot as plt 
    data = gpd.read_file(shapefile)
    landuselist = [' '] * polygoncount
    for index, row in data.iterrows():
        ''' SET the name of the column with land use information in the shapefile '''
        landuselist[index] = data.loc[index, 'LU_DESC'] 
    return landuselist



def findfixedID(basiclanduse, fixedlandtype):
    ''' find the unchangeable polygons' id based on the basic landuse '''
    #the __basiclanduse__ is a list of the original land use allocation
    #the __fixedlandtype__ is a list of unchangeable land use type
    #finally, it will return a list of the polygon IDs that should be unchangeable 

    fixedID = []
    for idx, val in enumerate(basiclanduse):
        if val in fixedlandtype:
            fixedID.append([idx, val])
    return fixedID

    
def readarealist(shapefile, polygoncount):
    ''' read the area of each polygon '''
    #the __shapefile__ is a shapefile consisting of the polygons need to be allocated
    #the __polygoncont__ is the number of polygons need to be allocated
    #finally, it will return a list of all the area of all the polygons

    import geopandas as gpd
    import matplotlib.pyplot as plt 
    data = gpd.read_file(shapefile)
    arealist = [' '] * polygoncount
    for index, row in data.iterrows():
        arealist[index] = data.loc[index, 'geometry'].area 
    return arealist


def landuse_ratio(landuselist, arealist, landusetype):
    ''' calculate the area ratio of a land use type '''
    #the __landuselist__ is the list of land use allocation information
    #the __arealist__ is the list of area of all the polygons
    #the __landusetyoe__ is the string of the land use type whose area ratio need to be calculated
    #finally, it will return a value of the area ratio of the land use type in the land use allocation

    area = 0
    for i in range(len(landuselist)):
        if landuselist[i] == landusetype:
            area += arealist[i]
    area_sum = sum(arealist)
    return area/area_sum

def landuse_ratio_tensor(landuselist, arealist, landusetype):
    ''' calculate the area ratio of a land use type '''
    #the __landuselist__ is the list of land use allocation information
    #the __arealist__ is the list of area of all the polygons
    #the __landusetyoe__ is the string of the land use type whose area ratio need to be calculated
    #finally, it will return a value of the area ratio of the land use type in the land use allocation
    landusetype = landusetype.unsqueeze(1)
    mask = landuselist == landusetype.expand(-1, landuselist.size(1))

    area_ratio = torch.sum(arealist * mask, dim=1)  # Sum over the num_landuses dimension
    return area_ratio

def initiallanduse_r(landtype, fixedID, neighbourlist, arealist):
    ''' initialize landuse '''  
    #the __landtype__ is the list of all the land use types for calculating compatibility
    #the __fixedID__ is the list of the ID of unchangeable polygons 
    #the __neighbourlist__ is the list of all the neighbour relationships between all the polygons
    ''' SET the __surrounding_radius__ to initialize the land use allocation in an assigned size of circle '''
    #the __arealist__ is the list of area of all the polygons
    #finally, it will return a list of land use allocation

    import string
    import random
    from copy import deepcopy  
    
    ''' SET the constraint of the land use area '''     
    Residential_ratio_basic = 0.5 
    Commercial_ratio_basic = 0.1 
    Office_ratio_basic = 0.15
    Education_ratio_basic = 0.02
    realOffice_ratio_basic = 0.05
    realResidential_ratio_basic = 0.2 
    realCommercial_ratio_basic = 0.05
    
    overall_area = sum(arealist)
   
    initlanduse = [' '] * len(neighbourlist)   
        
    initcenterlist = list(range(len(neighbourlist)))
    
    for info in fixedID:
        initlanduse[info[0]] = info[1]
        initcenterlist.remove(info[0])
        
    # define 'Residential'
    realResidential = 0
    while realResidential < (realResidential_ratio_basic * overall_area):
        center = random.choice(initcenterlist)
        initcenterlist.remove(center)
        initlanduse[center] = 'Residential'
        realResidential += arealist[center]

        
    # define 'Commercial'
    realCommercial = 0
    while realCommercial < (realCommercial_ratio_basic * overall_area):
        center = random.choice(initcenterlist)
        initcenterlist.remove(center)
        initlanduse[center] = 'Commercial'
        realCommercial += arealist[center]

    
    # define 'Office'
    realOffice = 0
    while realOffice < (realOffice_ratio_basic * overall_area):
        center = random.choice(initcenterlist)
        initcenterlist.remove(center)
        initlanduse[center] = 'Office'
        realOffice += arealist[center]

    
         
    # define 'Residential&Commercial'
    Residential = realResidential
    Commercial = realCommercial
    while Commercial < (Commercial_ratio_basic * overall_area):
        center = random.choice(initcenterlist)
        initcenterlist.remove(center)
        initlanduse[center] = 'Residential&Commercial'
        Commercial += arealist[center]
        Residential += arealist[center]


    # define 'SOHO'
    Office = realOffice
    while Office < (Office_ratio_basic * overall_area):
        center = random.choice(initcenterlist)
        initcenterlist.remove(center)
        initlanduse[center] = 'SOHO'
        Office += arealist[center]
        Residential += arealist[center]

            
    # add 'Residential' or 'Residential&Commercial' or 'SOHO' to make Residential_ratio > 0.5
    while Residential < (Residential_ratio_basic * overall_area):
        if initcenterlist != []:
            center = random.choice(initcenterlist)
            initcenterlist.remove(center)
            Residential_temp = random.choice(['Residential','Residential&Commercial','SOHO'])
            initlanduse[center] = Residential_temp
            Residential += arealist[center]
        else:
            return initiallanduse_r(landtype, fixedID, neighbourlist, arealist)   
            
    # define 'Education'
    Education = 0
    while Education < (Education_ratio_basic * overall_area):
        if initcenterlist != []:
            center = random.choice(initcenterlist)
            initcenterlist.remove(center)
            initlanduse[center] = 'Education'
            Education += arealist[center]

        else:
            return initiallanduse_r(landtype, fixedID, neighbourlist, arealist)
            
    # randomly define the undefined polygon
    # in our study, we will not plan more hospital land use
    initlandtype = deepcopy(landtype)
    initlandtype.remove('Hospital')
    for polygon in range(len(initlanduse)):
        if initlanduse[polygon] == ' ' :
            initlanduse[polygon] = deepcopy(random.choice(initlandtype))
            
    return initlanduse

def initiallanduse(landtype, fixedID, neighbourlist, surrounding_radius, arealist):
    ''' initialize landuse '''  
    #the __landtype__ is the list of all the land use types for calculating compatibility
    #the __fixedID__ is the list of the ID of unchangeable polygons 
    #the __neighbourlist__ is the list of all the neighbour relationships between all the polygons
    ''' SET the __surrounding_radius__ to initialize the land use allocation in an assigned size of circle '''
    #the __arealist__ is the list of area of all the polygons
    #finally, it will return a list of land use allocation

    import string
    import random
    from copy import deepcopy  
    
    ''' SET the constraint of the land use area '''     
    Residential_ratio_basic = 0.5 
    Commercial_ratio_basic = 0.1 
    Office_ratio_basic = 0.15
    Education_ratio_basic = 0.02
    realOffice_ratio_basic = 0.05
    realResidential_ratio_basic = 0.2 
    realCommercial_ratio_basic = 0.05
    
    overall_area = sum(arealist)
   
    initlanduse = [' '] * len(neighbourlist)   
        
    initcenterlist = list(range(len(neighbourlist)))
    
    for info in fixedID:
        initlanduse[info[0]] = info[1]
        initcenterlist.remove(info[0])
        
    # define 'Residential'
    realResidential = 0
    while realResidential < (realResidential_ratio_basic * overall_area):
        center = random.choice(initcenterlist)
        initcenterlist.remove(center)
        initlanduse[center] = 'Residential'
        realResidential += arealist[center]
        centerlist = [center]
        temp = []
        for circles in range(surrounding_radius):
            temp += centerlist
            centerlist_temp = []
            for subcenter in centerlist:
                surroundings = deepcopy(neighbourlist[subcenter][1:])            
                for polygon in surroundings:
                    if initlanduse[polygon] == ' ':
                        initcenterlist.remove(polygon)
                        initlanduse[polygon] = 'Residential'
                        realResidential += arealist[polygon]
                centerlist_temp += deepcopy(surroundings)
            centerlist_temp = list(set(centerlist_temp) - set(deepcopy(temp)))
            centerlist = (deepcopy(centerlist_temp)) 
        
    # define 'Commercial'
    realCommercial = 0
    while realCommercial < (realCommercial_ratio_basic * overall_area):
        center = random.choice(initcenterlist)
        initcenterlist.remove(center)
        initlanduse[center] = 'Commercial'
        realCommercial += arealist[center]
        centerlist = [center]
        temp = []
        for circles in range(surrounding_radius):
            temp += centerlist
            centerlist_temp = []
            for subcenter in centerlist:
                surroundings = deepcopy(neighbourlist[subcenter][1:])            
                for polygon in surroundings:
                    if initlanduse[polygon] == ' ':
                        initcenterlist.remove(polygon)
                        initlanduse[polygon] = 'Commercial'
                        realCommercial += arealist[polygon]
                centerlist_temp += deepcopy(surroundings)
            centerlist_temp = list(set(centerlist_temp) - set(deepcopy(temp)))
            centerlist = (deepcopy(centerlist_temp))
    
    # define 'Office'
    realOffice = 0
    while realOffice < (realOffice_ratio_basic * overall_area):
        center = random.choice(initcenterlist)
        initcenterlist.remove(center)
        initlanduse[center] = 'Office'
        realOffice += arealist[center]
        centerlist = [center]
        temp = []
        for circles in range(surrounding_radius):
            temp += centerlist
            centerlist_temp = []
            for subcenter in centerlist:
                surroundings = deepcopy(neighbourlist[subcenter][1:])            
                for polygon in surroundings:
                    if initlanduse[polygon] == ' ':
                        initcenterlist.remove(polygon)
                        initlanduse[polygon] = 'Office'
                        realOffice += arealist[polygon]
                centerlist_temp += deepcopy(surroundings)
            centerlist_temp = list(set(centerlist_temp) - set(deepcopy(temp)))
            centerlist = (deepcopy(centerlist_temp))  
    
         
    # define 'Residential&Commercial'
    Residential = realResidential
    Commercial = realCommercial
    while Commercial < (Commercial_ratio_basic * overall_area):
        center = random.choice(initcenterlist)
        initcenterlist.remove(center)
        initlanduse[center] = 'Residential&Commercial'
        Commercial += arealist[center]
        Residential += arealist[center]
        centerlist = [center]
        temp = []
        for circles in range(surrounding_radius):
            temp += centerlist
            centerlist_temp = []
            for subcenter in centerlist:
                surroundings = deepcopy(neighbourlist[subcenter][1:])            
                for polygon in surroundings:
                    if initlanduse[polygon] == ' ':
                        initcenterlist.remove(polygon)
                        initlanduse[polygon] = 'Residential&Commercial'
                        Commercial += arealist[polygon]
                        Residential += arealist[polygon]
                centerlist_temp += deepcopy(surroundings)
            centerlist_temp = list(set(centerlist_temp) - set(deepcopy(temp)))
            centerlist = (deepcopy(centerlist_temp)) 

    # define 'SOHO'
    Office = realOffice
    while Office < (Office_ratio_basic * overall_area):
        center = random.choice(initcenterlist)
        initcenterlist.remove(center)
        initlanduse[center] = 'SOHO'
        Office += arealist[center]
        Residential += arealist[center]
        centerlist = [center]
        temp = []
        for circles in range(surrounding_radius):
            temp += centerlist
            centerlist_temp = []
            for subcenter in centerlist:
                surroundings = deepcopy(neighbourlist[subcenter][1:])            
                for polygon in surroundings:
                    if initlanduse[polygon] == ' ':
                        initcenterlist.remove(polygon)
                        initlanduse[polygon] = 'SOHO'
                        Office += arealist[polygon]
                        Residential += arealist[polygon]
                centerlist_temp += deepcopy(surroundings)
            centerlist_temp = list(set(centerlist_temp) - set(deepcopy(temp)))
            centerlist = (deepcopy(centerlist_temp)) 
            
    # add 'Residential' or 'Residential&Commercial' or 'SOHO' to make Residential_ratio > 0.5
    while Residential < (Residential_ratio_basic * overall_area):
        if initcenterlist != []:
            center = random.choice(initcenterlist)
            initcenterlist.remove(center)
            Residential_temp = random.choice(['Residential','Residential&Commercial','SOHO'])
            initlanduse[center] = Residential_temp
            Residential += arealist[center]
            centerlist = [center]
            temp = []
            for circles in range(surrounding_radius):
                temp += centerlist
                centerlist_temp = []
                for subcenter in centerlist:
                    surroundings = deepcopy(neighbourlist[subcenter][1:])            
                    for polygon in surroundings:
                        if initlanduse[polygon] == ' ':
                            initcenterlist.remove(polygon)
                            initlanduse[polygon] = Residential_temp
                            Residential += arealist[polygon]
                    centerlist_temp += deepcopy(surroundings)
                centerlist_temp = list(set(centerlist_temp) - set(deepcopy(temp)))
                centerlist = (deepcopy(centerlist_temp)) 
        else:
            return initiallanduse(landtype, fixedID, neighbourlist, 2, arealist)   
            
    # define 'Education'
    Education = 0
    while Education < (Education_ratio_basic * overall_area):
        if initcenterlist != []:
            center = random.choice(initcenterlist)
            initcenterlist.remove(center)
            initlanduse[center] = 'Education'
            Education += arealist[center]
            centerlist = [center]
            temp = []
            for circles in range(surrounding_radius):
                temp += centerlist
                centerlist_temp = []
                for subcenter in centerlist:
                    surroundings = deepcopy(neighbourlist[subcenter][1:])            
                    for polygon in surroundings:
                        if initlanduse[polygon] == ' ':
                            initcenterlist.remove(polygon)
                            initlanduse[polygon] = 'Education'
                            Education += arealist[polygon]
                    centerlist_temp += deepcopy(surroundings)
                centerlist_temp = list(set(centerlist_temp) - set(deepcopy(temp)))
                centerlist = (deepcopy(centerlist_temp)) 
        else:
            return initiallanduse(landtype, fixedID, neighbourlist, 2, arealist)
            
    # randomly define the undefined polygon
    # in our study, we will not plan more hospital land use
    initlandtype = deepcopy(landtype)
    initlandtype.remove('Hospital')
    for polygon in range(len(initlanduse)):
        if initlanduse[polygon] == ' ' :
            initlanduse[polygon] = deepcopy(random.choice(initlandtype))
            
    return initlanduse



def visuallanduse(shapefile, landuselist, landusePalette, figname,titlestr = "title"):
    ''' visualize landuse '''
    #the __shapefile__ is the shapefile with land use allocation information, which will be applied in the programme 
    #the __landuselist__ is the the list of the land use attribution, which need to be visualized  
    ''' SET __landusePalette__ as a dictionary to assign each land use type a particular color  '''
    ''' SET __figname__ as a string to assign layout figure a path and a name  '''
    #finally, it will return a figure showing the land use allocation
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.patches as mpatches 
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    import matplotlib.font_manager as fm
    # load data and add landuse list to attribute table
    data0 = gpd.read_file(shapefile)
    for index, row in data0.iterrows():
        data0.loc[index, 'landuse'] = landuselist[index]
    # Plot data
    fig, ax = plt.subplots(figsize=(10, 8))
    # Loop through each attribute type and plot it using the colors assigned in the dictionary
    patch = []
    for ctype, data in data0.groupby('landuse'):
    # Define the color for each group using the dictionary
        color = landusePalette[ctype]
    # Plot each group using the color defined above
        data.plot(color=color, ax=ax)
    # Prepare for drawing legend
        patch.append(mpatches.Patch(color=color, label=ctype))
    plt.title(titlestr, fontsize=35, fontproperties='SimHei')
    # Plot Scaler Bar   
    fontprops = fm.FontProperties(size=14)
    scalebar = AnchoredSizeBar(ax.transData,
                           100, '100 m', loc='upper right', 
                           pad=6,
                           sep=5,
                           color='black',
                           frameon=False,
                           size_vertical=1,
                           fontproperties=fontprops)
    ax.add_artist(scalebar)
    
    # Plot legend
    ax.legend(handles=patch, loc='lower left', fontsize=14, frameon=False)

    # Plot North Arrow
    lon0, lon1 = ax.get_xlim()
    lat0, lat1 = ax.get_ylim()
    ax.text(lon1-(lon1-lon0)*0.165, lat1-(lat1-lat0)*0.12, u'\u25B2 \nN ', ha='center', fontsize=30, family='Arial', rotation = 0)
           
    ax.set_axis_off()    
    plt.tight_layout()
    plt.show()
    plt.pause(1)
    plt.close()
    # plt.savefig(figname, dpi=300)



def calCompatibility(landtype, neighbourlist, landuselist, CompatibilityTable):
    ''' calculate Compatibility '''
    #the __landtype__ is the list of all the land use types for calculating compatibility
    #the __neighbourlist__ is the list of all the neighbour relationships between all the polygons
    #the __landuselist__ is the the list of the land use attribution, whose compatibility need to be calculated  
    #the __CompatibilityTable__ is the list of all the compatibility scores between all the land use types
    #finally, it will return a value of compatibility  
  
    ComList = []
    # create a list to list the Compatibility of each center (all the polygons)
    count = 0
    for center in neighbourlist:
    # in the 'neighbourlist' (2 dimension list), center = neighbourlist[i]
    
    # for each center:
    #### neighbourlist.index(center) = the center's ID
    #### center[0] = the count of the center's neighbours
    #### center[1:] = the IDs of the center's neighbours
        
        
        com = 0
        # for each center, start to calculate its Compatibility with creating a variable 'com' = 0 
        
        'get the landuse type of the center'
        center_landuse = landuselist[neighbourlist.index(center)]
        # for each value in the 'landuselist' (1 dimension list):
        #### the index of the value = the ID of the polygon
        #### the value = the landuse type of the polygon
        
        for neighbour in center[1:]:
        # for each neighbour of the center
            
            'get the landuse type of the neighbour'
            neighbour_landuse = landuselist[neighbour]
            
            ' the Compatibility of each center(polygon) = sum(the Compatibility of the center & each of its neighbours) '
            com += CompatibilityTable[landtype.index(center_landuse)][landtype.index(neighbour_landuse)]
            count += 1
            # in the 'Compatibility Table' (2 dimension list),
            #### the index of row and column = the index of 'landtype' (1 dimension list)  
            #### represent that the value located at that row and column = the Compatibility of that two landuse types
            
        ComList.append(com)
        # list the Compatibility of each center (all the polygons)
    
    ' Finally, the Compatibility of a landuselist = sum(the Compatibility of each center(polygon))  ' 
    return sum(ComList)/count

def calculate_distance_matrix(locs):
    num_points = len(locs)
    # 初始化距离矩阵，使用无穷大填充，以避免除以零的错误
    distance_matrix = np.zeros((num_points, num_points))

    # 计算点之间的欧几里得距离
    for i in range(num_points):
        for j in range(i + 1, num_points):  # 只需计算上三角矩阵，其余部分通过对称性获得
            x1, y1 = locs[i]
            x2, y2 = locs[j]
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance  # 由于矩阵是对称的，复制到下三角部分

    return distance_matrix
def normalizeloc(shapefile):
    import geopandas as gpd
    import math
    data = gpd.read_file(shapefile)
    centroidseries = data['geometry'].centroid
    locs = []
    for i in range(len(centroidseries)):
        locs.append([centroidseries[i].x, centroidseries[i].y])
    minx = min([loc[0] for loc in locs])
    miny = min([loc[1] for loc in locs])
    maxx = max([loc[0] for loc in locs])
    maxy = max([loc[1] for loc in locs])
    for i in range(len(locs)):
        locs[i][0] = (locs[i][0] - minx) / (maxx - minx)
        locs[i][1] = (locs[i][1] - miny) / (maxy - miny)
    return locs

def normalizearea(arealist):
    total_area = sum(arealist)
    return [area / total_area for area in arealist]

def calCompactness(neighbourlist, landuselist):
    ''' calculate compactness '''
    #the __neighbourlist__ is the list of all the neighbour relationships between all the polygons
    #the __landuselist__ is the the list of the land use attribution, whose compatibility need to be calculated  
    
    temp = [[' ' for x in range(1)] for y in range(len(neighbourlist))]
    # use a temp to find all the cluster of a landuselist
    
    "calculte the compactness by calculating the cluster's count"
    clustercount = 0
    
    "for each center, find the 1-radius cluster(only its neighbours) around it"
    for center in neighbourlist:
        for neighbour in center[1:]:
        # for each neighbour of all the centers
        
            if landuselist[neighbour] == landuselist[neighbourlist.index(center)]:
            # if the landuse type of the neighbour = the landuse type of the center
            
                temp[neighbourlist.index(center)][0] = neighbourlist.index(center)
                # in the temp list, temp[][0] = the ID of the center
                
                temp[neighbourlist.index(center)].append(neighbour) 
                # temp[][1:] = the ID of the neighbours whose landuse type is the same with the center
    
    "calculate the cluster's count by combining the 1-radius clusters with intersection " 
    for count in range(2): 
    # twice Bubble Sort to make sure all the clusters do not have intersection with each other 
    
        'find the clusters with intersection refers to Bubble Sort'
        for i in range(len(temp)):
        # for each center        
            if temp[i] != []: 
            # if the center do have a neighbour with the same landuse type (1-radius cluster)
                
                for j in range(i+1,len(temp)):
                # compare the 1-radius cluter between the center and the uncompared 1-radius clusters   
                
                    if temp[i] != [' '] and len(set(temp[i])&set(temp[j])) > 0:
                    # if the two 1-radius clusters have intersection
                        temp[i] = list(set(temp[i] + temp[j]))
                        # combine both of them into one cluster and replace the 1-radius cluster of the center 
                        temp[j] = []
                        # delete the compared 1-radius cluster 
                        
                if count == 1:
                # at the second times to Bubble Sort, the clusters do not have intersection with other clusters  
                    clustercount += 1   
                    # regard these clusters as the ture clusters, and count the clusters of the landuselist
                    
    "Finally, the Compactness of a landuselist = - the cluster's count"                
    return - clustercount
                

   
def calInterDistance(shapefile, polygoncount):
    ''' calculate Internal Distance between all the polygons ''' 
    #the __shapefile__ is the shapefile with land use allocation information, which will be applied in the programme 
    #the __polygoncont__ is the number of polygons need to be allocated

    import geopandas as gpd
    import math
    data = gpd.read_file(shapefile)
    centroidseries = data['geometry'].centroid
    interDistanceList = [[0 for x in range(polygoncount)] for y in range(polygoncount)]
    i=0
    for pointA in centroidseries:
        j=0
        for pointB in centroidseries:    
            interDistanceList[i][j] = math.sqrt((pointB.x-pointA.x)**2 + (pointB.y-pointA.y)**2)
            j += 1
        i +=1
    return interDistanceList



def calInterAccessibility_Average(landuselist, landtypeA, landtypeB, interDistanceList):
    ''' calculate Internal Accessibility between two landtype '''
    #the __landuselist__ is the the list of the land use attribution, whose accessibility need to be calculated 
    #the __landtypeA__ is the string of one or more than one land use type for one side of accessibility
    #the __landtypeB__ is the string of one or more than one land use type for another side of accessibility
    #the __interDistance__ is the list of all the distances between all the polygons 
    #finally, it will return a value of internal accessibility

    AllDistance = 0 
    count = 1
    for i in range(len(landuselist)):
        if landuselist[i] == landtypeA:
            for j in range(len(landuselist)):
                if landuselist[j] == landtypeB:
                    Distance = interDistanceList[i][j]
                    AllDistance += Distance
                    count += 1  
    AverDistance = AllDistance/count
    return -AverDistance

def mutaion_random(polygoncount, landtype, fixedID, num_MutationCenter, arealist, parent):
    
    import random
    import init
    from copy import deepcopy
    

    centerlist = list(range(polygoncount))    
    for info in fixedID:
        centerlist.remove(info[0])
    
    p_landuselist = deepcopy(parent)
    offsprings_temp = deepcopy(p_landuselist)
    centers = random.sample(centerlist, num_MutationCenter)
    for center in centers:
        center_landuse = deepcopy(p_landuselist[center])
        other_landuse = deepcopy(landtype)
        other_landuse.remove('Hospital')
#            print(center_landuse,other_landuse)
        other_landuse.remove(center_landuse)
        offsprings_temp[center] = random.choice(other_landuse)
    Residential_ratio = init.landuse_ratio(offsprings_temp, arealist, 'Residential') 
    Commercial_ratio = init.landuse_ratio(offsprings_temp, arealist, 'Commercial') 
    Education_ratio = init.landuse_ratio(offsprings_temp, arealist, 'Education')
    Office_ratio = init.landuse_ratio(offsprings_temp, arealist, 'Office')
    SOHO_ratio = init.landuse_ratio(offsprings_temp, arealist, 'SOHO')
    RC_ratio = init.landuse_ratio(offsprings_temp, arealist, 'Residential&Commercial')
    if Residential_ratio+SOHO_ratio+RC_ratio < 0.5 or Commercial_ratio+RC_ratio < 0.1 or Education_ratio < 0.02 or Office_ratio+SOHO_ratio < 0.15 or Office_ratio < 0.05 or Residential_ratio < 0.2 or Commercial_ratio < 0.05:
#            print('again')
        return mutaion_random(polygoncount, landtype, fixedID, num_MutationCenter, arealist, parent)
    else:   
#            print('+1')
        return offsprings_temp  

def mutaion_boundary(neighbourlist, fixedID, num_MutationCenter, arealist, parent):
    
    import random
    import init
    from copy import deepcopy
    
    p_landuselist = deepcopy(parent)
    offsprings_temp = deepcopy(parent)
    differentlist = deepcopy(neighbourlist)
    centerlist = list(range(len(neighbourlist))) 
    for info in fixedID:
        centerlist.remove(info[0])  
    for center in range(len(neighbourlist)):
        differentlist[center][0] = center
        for neighbour in neighbourlist[center][1:]:
            if p_landuselist[center] == p_landuselist[neighbour] or p_landuselist[neighbour] == 'Hospital':
#                    print(center, neighbour)
                differentlist[center].remove(neighbour)
        if len(differentlist[center]) == 1:
            centerlist.remove(center)
                                     
    centers = random.sample(centerlist, num_MutationCenter)
    for center in centers:          
        neighbours = deepcopy(differentlist[center][1:])
        neighbours_landuse = []
        for n in neighbours:
            neighbours_landuse.append(p_landuselist[n])
        offsprings_temp[center] = deepcopy(random.choice(neighbours_landuse))
    
    
    Residential_ratio = init.landuse_ratio(offsprings_temp, arealist, 'Residential') 
    Commercial_ratio = init.landuse_ratio(offsprings_temp, arealist, 'Commercial') 
    Education_ratio = init.landuse_ratio(offsprings_temp, arealist, 'Education')
    Office_ratio = init.landuse_ratio(offsprings_temp, arealist, 'Office')
    SOHO_ratio = init.landuse_ratio(offsprings_temp, arealist, 'SOHO')
    RC_ratio = init.landuse_ratio(offsprings_temp, arealist, 'Residential&Commercial')
    if Residential_ratio+SOHO_ratio+RC_ratio < 0.5 or Commercial_ratio+RC_ratio < 0.1 or Education_ratio < 0.02 or Office_ratio+SOHO_ratio < 0.15 or Office_ratio < 0.05 or Residential_ratio < 0.2 or Commercial_ratio < 0.05:
        return mutaion_boundary(neighbourlist, fixedID, num_MutationCenter, arealist, parent) 
    else:
        return offsprings_temp 






