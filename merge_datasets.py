import os
import json
import shutil
import random
from pathlib import Path

def convert_labelme_to_yolo_seg(json_path, class_id):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    img_w = data['imageWidth']
    img_h = data['imageHeight']
    
    yolo_lines = []
    for shape in data['shapes']:
        points = shape['points']
        # Flatten and normalize points
        normalized_points = []
        for p in points:
            x = p[0] / img_w
            y = p[1] / img_h
            normalized_points.extend([f"{x:.6f}", f"{y:.6f}"])
        
        line = f"{class_id} " + " ".join(normalized_points)
        yolo_lines.append(line)
        
    return yolo_lines

def main():
    base_dir = Path(r"c:\Users\carol\testee\Detector_objetos_AI\DETECTOR")
    folders = {"platano": 0, "manzana": 1, "tetrapack": 2}
    
    out_dir = base_dir / "dataset_unified"
    if out_dir.exists():
        shutil.rmtree(out_dir)
        
    out_dir.mkdir()
    
    for split in ['train', 'val']:
        (out_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (out_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
    all_data = []
    
    for folder_name, class_id in folders.items():
        folder_path = base_dir / folder_name
        if not folder_path.exists():
            print(f"Directory {folder_path} does not exist.")
            continue
            
        for file in os.listdir(folder_path):
            if file.endswith('.json'):
                json_path = folder_path / file
                # Find matching image
                img_name = None
                for ext in ['.jpg', '.jpeg', '.png']:
                    possible_img = json_path.with_suffix(ext)
                    if possible_img.exists():
                        img_name = possible_img.name
                        break
                        
                if img_name:
                    all_data.append({
                        "json_path": json_path,
                        "img_path": folder_path / img_name,
                        "class_id": class_id,
                        "folder_name": folder_name
                    })

    # Shuffle and split
    random.seed(42)
    random.shuffle(all_data)
    
    split_idx = int(len(all_data) * 0.8)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    def process_data(data_list, split):
        for item in data_list:
            # unique prefix to avoid filename collisions
            prefix = item['folder_name'] + "_"
            new_img_name = prefix + item['img_path'].name
            new_txt_name = prefix + item['json_path'].stem + ".txt"
            
            # Convert
            yolo_lines = convert_labelme_to_yolo_seg(item['json_path'], item['class_id'])
            
            # Write txt
            out_txt_path = out_dir / split / 'labels' / new_txt_name
            with open(out_txt_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(yolo_lines))
                
            # Copy image
            out_img_path = out_dir / split / 'images' / new_img_name
            shutil.copy2(item['img_path'], out_img_path)

    process_data(train_data, 'train')
    process_data(val_data, 'val')
    
    # Create dataset.yaml
    yaml_content = f"""path: {out_dir.absolute()}
train: train/images
val: val/images

nc: 3
names: ["Platano", "Manzana", "Tetrapack"]
"""
    yaml_path = out_dir / "dataset.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
        
    print(f"Successfully processed {len(all_data)} files.")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    print(f"Dataset created at {out_dir}")

if __name__ == "__main__":
    main()
