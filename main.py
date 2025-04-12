import matplotlib
matplotlib.use('Qt5agg')
import matplotlib.pyplot as plt
import torch
from PIL import Image  
from transformers import DPTForDepthEstimation , DPTImageProcessor
import numpy as np
import open3d as o3d

feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

image = Image.open("test3.jpg").convert("RGB")
image = image.resize((384, 384))
new_height = 480 if image.height > 480 else image.height
new_height -= (new_height % 32)
new_width = int(new_height * image.width / image.height)
diff = new_width % 32

new_width = new_width + diff if diff < 16 else new_width + 32 - diff
new_size = (new_width, new_height)
image = image.resize(new_size)

inputs = feature_extractor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    print(outputs)
    predicted_depth = outputs.predicted_depth

pad = 16
output = predicted_depth.squeeze().cpu().numpy() * 1000.0
output = output[pad:-pad, pad:-pad]
image = image.crop((pad, pad, image.width - pad, image.height - pad))

fig , ax = plt.subplots(1,2,figsize=(10,5))
ax[0].imshow(image)
ax[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
ax[1].imshow(output, cmap='plasma')
ax[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
plt.tight_layout()    
plt.pause(10)

width, height = image.size

depth_image = (output / np.max(output) * 1000).astype(np.uint16)  
image = np.array(image)

depth_o3d = o3d.geometry.Image(depth_image)
image_o3d = o3d.geometry.Image(image)
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, depth_o3d, convert_rgb_to_intensity=False)



fx, fy = width / 2, height / 2 
cx, cy = width / 2.5, height / 2.5
camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

pcd_raw = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
# o3d.visualization.draw_geometries([pcd_raw])

c1,ind = pcd_raw.remove_statistical_outlier(nb_neighbors=50,std_ratio=1.5)
pcd = pcd_raw.select_by_index(ind)

pcd = pcd.voxel_down_sample(voxel_size=0.001)

pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=60))  
pcd.orient_normals_consistent_tangent_plane(k=12)
o3d.visualization.draw_geometries([pcd])

mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth = 10, n_threads = 1)[0]

rotation = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
mesh.rotate(rotation, center=(0,0,0))

o3d.visualization.draw_geometries([mesh] , mesh_show_back_face = True)