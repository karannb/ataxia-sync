import os
import json
import numpy as np
from tqdm import tqdm

def extractNormal(video):
    
    ovr_kpts = []
        
        
    for frame in range(1, 181):
        if not os.path.exists(f"data/keypoints_norm_v3/{video}/output_{frame:05d}.json"):
            continue
        
        with open(f"data/keypoints_norm_v3/{video}/output_{frame:05d}.json") as f:
            data = json.load(f)
            
        if frame == 1:
            keypoints = np.array(data["people"][0]["pose_keypoints_2d"]).reshape(-1, 3)
            ovr_kpts.append(keypoints)
            continue
        
        else:
            
            min_person = None
            min_dist = 1e9
            
            for i, person in enumerate(data["people"]):
                
                keypoints = np.array(person["pose_keypoints_2d"]).reshape(-1, 3)
                dist = np.linalg.norm(keypoints[:, :2] - ovr_kpts[-1][:, :2])
                
                if dist < min_dist:
                    min_dist = dist
                    min_person = i
            
            if min_person is None:
                print(f"Frame {frame} of {video} has no person")
                continue  
            keypoints = np.array(data["people"][min_person]["pose_keypoints_2d"]).reshape(-1, 3)
            ovr_kpts.append(keypoints) 
                
    kpts = np.array(ovr_kpts)
    
    np.save(f"data/keypoints_ver6/{video}.npy", kpts)
    
    return kpts

if __name__ == "__main__":
    
    for video in tqdm(os.listdir("data/keypoints/")):
        
        extractNormal(video)
    
    print("Done!")