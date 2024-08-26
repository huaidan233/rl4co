import random

import torch
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import colormaps

from rl4co.envs.urbanplan.cityplan import init
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)
landusePalette = {'Commercial': 'coral',
           'Residential': 'peachpuff',
           'Office': 'indianred',
           'Residential&Commercial': 'lightsalmon',
           'Green Space': 'lightgreen',
           'Education': 'lightskyblue',
           'Hospital': 'royalblue',
           'SOHO': 'lightcoral'
           }
def render(td, actions=None,ax=None,planout=None):
    if ax is None:
        # Create a plot of the nodes
        fig, ax = plt.subplots(figsize=(10, 6))
    landtype = ['Commercial', 'Residential', 'Office', 'Residential&Commercial', 'Green Space', 'Education', 'Hospital',
                'SOHO']
    td = td.detach().cpu()

    # If batch_size greater than 0 , we need to select the first batch element
    if td.batch_size != torch.Size([]):
        td = td[0]
        actions = actions[0]
    if actions is None:
        actions = td.get("action", None)
    locs = td["locs"]
    current_plan = td["current_plan"].tolist()
    areaslist = np.array(td["areas"].tolist())
    x, y = locs[:, 0].tolist(), locs[:, 1].tolist()
    actions = actions.tolist()
    latest_plan = current_plan
    select_type = calc_next_type(latest_plan, landtype, areaslist)
    for action in actions:
        if action is not None and select_type is not None:
            latest_plan[action] = select_type.item()
            select_type = calc_next_type(latest_plan, landtype, areaslist)
        else:
            break
    strplan = init.map_to_strings(latest_plan, landtype)
    plan_colors = [landusePalette.get(p, 'gray') for p in strplan]

    # Plot the visited nodes with colors based on current_plan and sizes based on areas
    ax.scatter(x, y, color=plan_colors, s=areaslist * 5000)  # Adjust the scale as needed

    # Add arrows between visited nodes as a quiver plot
    # dx, dy = np.diff(x), np.diff(y)
    # ax.quiver(x[:-1], y[:-1], dx, dy, scale_units="xy", angles="xy", scale=1, color="k")
    # 添加颜色图例并将其放置在右侧
    handles = [plt.Line2D([0], [0], marker='o', color=color, markersize=10, label=landtype)
               for landtype, color in landusePalette.items()]
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5), title='Land Use Type')

    # Setup limits and show
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    plt.subplots_adjust(right=0.75)  # 调整右边界，给图例留出空间
    plt.show()
    if planout == True:
        return init.map_to_strings(latest_plan, landtype)


def calc_next_type(current_plan, landtypes, areas):
    strstate = init.map_to_strings(current_plan, landtypes)
    Residential_ratio = init.landuse_ratio(strstate, areas, 'Residential')
    Commercial_ratio = init.landuse_ratio(strstate, areas, 'Commercial')
    Education_ratio = init.landuse_ratio(strstate, areas, 'Education')
    Office_ratio = init.landuse_ratio(strstate, areas, 'Office')
    SOHO_ratio = init.landuse_ratio(strstate, areas, 'SOHO')
    RC_ratio = init.landuse_ratio(strstate, areas, 'Residential&Commercial')
    # 对每个批次进行计算
    if Education_ratio < 0.02:
        type = 'Education'
        return find_type_index(type)
    if Residential_ratio < 0.2:
        type = 'Residential'
        return find_type_index(type)
    if Commercial_ratio + RC_ratio < 0.1:
        type = 'Residential&Commercial'
        return find_type_index(type)
    if Commercial_ratio < 0.05:
        type = 'Commercial'
        return find_type_index(type)
    if Office_ratio < 0.05:
        type = 'Office'
        return find_type_index(type)
    if Office_ratio + SOHO_ratio < 0.15:
        type = 'SOHO'
        return find_type_index(type)
    if Residential_ratio + SOHO_ratio + RC_ratio < 0.5:
        type = random.choices(['Residential', 'SOHO', 'Residential&Commercial'], [0.3, 0.3, 0.4])[0]
        return find_type_index(type)
    else:
        return
def find_type_index(search_type):
    landtype = ['Commercial', 'Residential', 'Office', 'Residential&Commercial', 'Green Space', 'Education', 'Hospital',
                'SOHO']
    landtype_array = np.array(landtype)
    index = np.where(landtype_array == search_type)[0]
    return index