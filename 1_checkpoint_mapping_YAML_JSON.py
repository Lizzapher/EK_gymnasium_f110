import os
import time
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image

#----------
# Variables
#----------
MAP_LIST = ['./maps/Catalunya/Catalunya_map', 
            './maps/Montreal/Montreal_map', 
            './maps/Monza/Monza_map', 
            './maps/Shanghai/Shanghai_map', 
            './maps/Spa/Spa_map', 
            './maps/YasMarina/YasMarina_map']
# _MAP = './maps/BrandsHatch/BrandsHatch_map'
_EXT = 'png'

#-----------------
# Plot Checkpoints
#-----------------
def world_to_map_coords(world_x, world_y, resolution, origin, img_height):
    world_x = float(world_x)
    world_y = float(world_y)
    resolution = float(resolution)
    origin_0 = float(origin[0])
    origin_1 = float(origin[1])

    map_x = float((world_x - origin_0) / resolution)
    map_y = img_height - (world_y - origin_1) / resolution
    return map_x, map_y

def plot_checkpoints_on_map(map_list):
    for map in range(len(map_list)):
        with open(f'{map_list[map]}.yaml', 'r') as f:
            map_data = yaml.safe_load(f)

        image_path = Path(f'{map_list[map]}.{_EXT}').parent / map_data['image']
        resolution = map_data['resolution']
        origin = map_data['origin']

        img = Image.open(image_path)
        img = img.convert("RGB")
        width, height = img.size

        with open(f'{map_list[map]}.json', 'r') as f:
            cp_data = json.load(f)

        cps = cp_data.get('checkpoints', [])
        if not cps:
            raise ValueError("No checkpoints found in JSON")

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)

        for idx, cp in enumerate(cps):
            # cp is [[x1,y1], [x2,y2]]
            (wx1, wy1), (wx2, wy2) = cp

            # convert both endpoints to pixel coords
            px1, py1 = world_to_map_coords(wx1, wy1, resolution, origin, height)
            px2, py2 = world_to_map_coords(wx2, wy2, resolution, origin, height)

            # draw the line segment
            ax.plot([px1, px2], [py1, py2],
                    linewidth=3, color='red', alpha=0.6)
            # label it at the midpoint
            mx, my = (px1 + px2) / 2, (py1 + py2) / 2
            ax.text(mx, my, str(idx),
                    color='black', fontsize=10,
                    ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.5, boxstyle='round'))

        ax.set_title("Track with Checkpoints")
        ax.axis('off')
        plt.tight_layout()
        out_path = f'{map_list[map]}_with_checkpoints.png'
        plt.savefig(out_path, dpi=300)
        print(f"Saved visualization to {out_path}")

#------------
# Main Method
#------------
def main():
    plot_checkpoints_on_map(MAP_LIST)

#-----------------------------------
# Main guard
#-----------------------------------
if __name__ == "__main__":
    main()