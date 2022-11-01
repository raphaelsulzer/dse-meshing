import os, sys
import time
from tqdm import tqdm
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

from logmap_estimation.eval_network import reconstruct
from logmap_alignment.align_patches import align_patch
from logmap_alignment.eval_align_meshes import align_meshes
from triangle_selection.selection import write_candidates

from multiprocessing import RawArray
import multiprocessing
import functools, trimesh

sys.path.append(os.path.join(os.path.dirname(__file__),"..","benchmark","datasets"))
from modelnet10 import ModelNet10
from shapenet import ShapeNet
from berger import Berger




dataset = ModelNet10()
split = "test"
models = dataset.getModels(splits=[split],classes=["bathtub", "bed", "desk", "dresser", "nightstand", "toilet"])[split]
outpath = "/mnt/raphael/ShapeNet_out/benchmark/dse/modelnet"



# dataset = Berger()
# models = dataset.getModels(type="berger")
# outpath = "/mnt/raphael/benchmark/dse/reconbench"

os.makedirs(outpath,exist_ok=True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
BBOX = 1000.0

global align_patch_func


def align_patch_func(shared_predicted_map, shared_predicted_neighborhood_indices, i):
    return align_patch(shared_predicted_map, shared_predicted_neighborhood_indices, i)

for m in models:

    cpath = os.path.join(outpath,m["class"])
    os.makedirs(cpath,exist_ok=True)
    dse_path = os.path.join(os.path.dirname(m["scan"]),"..","dse",m["class"])
    os.makedirs(dse_path,exist_ok=True)

    # logmap estimation
    logmap_model = os.path.join(ROOT_DIR, 'data/pretrained_models/pretrained_logmap/model.ckpt')
    classifier_model = os.path.join(ROOT_DIR, 'data/pretrained_models/pretrained_classifier/model.ckpt')
    reconstruct(m["scan"], classifier_model, logmap_model, dse_path)

    # logmap allignment
    predicted_map = np.load(os.path.join(dse_path,"predicted_map.npy"))

    points = np.load(m["scan"])["points"]
    corrected_maps = np.zeros_like(predicted_map)
    n_points = len(predicted_map)
    predicted_neighborhood_indices = np.load(os.path.join(dse_path,"predicted_neighborhood_indices.npy"))
    full_errors = []
    BATCH_SIZE = 64#512#16
    n_nearest_neighbors = 30
    # n_points = 3000
    shared_predicted_map = RawArray('d',n_points*(n_nearest_neighbors+1)*3)
    shared_predicted_map = np.frombuffer(shared_predicted_map, dtype=np.float64).reshape(n_points, (n_nearest_neighbors+1), 3)
    np.copyto(shared_predicted_map, predicted_map)

    shared_predicted_neighborhood_indices =RawArray('i',n_points*(n_nearest_neighbors+1))
    shared_predicted_neighborhood_indices = np.frombuffer(shared_predicted_neighborhood_indices, dtype=np.int32).reshape(n_points, (n_nearest_neighbors+1))
    np.copyto(shared_predicted_neighborhood_indices, predicted_neighborhood_indices)

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    # jobs = []
    print('start:', m["scan"])
    with multiprocessing.Pool(64) as pool:
        corrected_maps = pool.map(functools.partial(align_patch_func,shared_predicted_map,shared_predicted_neighborhood_indices), range(n_points))
        corrected_maps = np.array(corrected_maps)
        np.save(os.path.join(dse_path,'corrected_maps.npy'), corrected_maps)

    # align meshes
    align_meshes(m["scan"],dse_path)

    # selection
    file=m["scan"]
    print("triangle selection:",file)
    bbox_diag=write_candidates(file,dse_path)
    arg1 = os.path.join(dse_path, "pred.txt")
    arg2 = os.path.join(cpath, m["model"]+".ply")
    os.system(os.path.join(ROOT_DIR, "triangle_selection/postprocess/build/postprocess {} {}".format(arg1, arg2)))

    print("triangle resize:",file)
    arg2 = os.path.join(cpath, m["model"]+".ply")
    mesh = trimesh.load(arg2, process=False)
    mesh.vertices*=bbox_diag/BBOX
    trimesh.repair.fix_normals(mesh) # orient faces coherently
    mesh.export(arg2)



