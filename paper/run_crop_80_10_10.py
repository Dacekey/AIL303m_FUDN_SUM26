import os
import cv2
from collections import defaultdict
import random
from tqdm import tqdm
import shutil
import json

dataset_dir = "paper/dataset_NVTS"
images_dir = os.path.join(dataset_dir, "images")
labels_dir = os.path.join(dataset_dir, "labels")
classes_file = os.path.join(dataset_dir, "classes.txt")

output_dir = "paper/cropped_dataset"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "val")
test_dir = os.path.join(output_dir, "test")

with open(classes_file, "r") as f:
    classes = [line.strip() for line in f.readlines()]

class_image_counts = defaultdict(int)
for filename in tqdm(os.listdir(labels_dir), desc="Counting classes"):
    if not filename.endswith(".txt"): continue
    filepath = os.path.join(labels_dir, filename)
    try:
        with open(filepath, "r") as f:
            class_ids = set()
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_ids.add(int(parts[0]))
            for cid in class_ids:
                class_image_counts[cid] += 1
    except Exception:
        pass

valid_classes_idx = {cid for cid, count in class_image_counts.items() if count >= 20}
valid_classes_names = {cid: classes[cid] for cid in valid_classes_idx}

class_crops = defaultdict(list)
for filename in tqdm(os.listdir(labels_dir), desc="Parsing labels"):
    if not filename.endswith(".txt"): continue
    img_filename = filename.replace(".txt", ".jpg")
    if not os.path.exists(os.path.join(images_dir, img_filename)): continue
    
    with open(os.path.join(labels_dir, filename), "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cid = int(parts[0])
                if cid in valid_classes_idx:
                    class_crops[cid].append((img_filename, float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for cid, crops in tqdm(class_crops.items(), desc="Cropping images"):
    random.seed(42)
    random.shuffle(crops)
    
    n = len(crops)
    train_idx = int(n * 0.8)
    val_idx = int(n * 0.9)
    
    train_crops = crops[:train_idx]
    val_crops = crops[train_idx:val_idx]
    test_crops = crops[val_idx:]
    
    class_name_safe = valid_classes_names[cid].replace("/", "_").replace("*", "_")
    os.makedirs(os.path.join(train_dir, class_name_safe), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name_safe), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name_safe), exist_ok=True)
    
    def process_crops(crop_list, split_name):
        for i, (img_name, x, y, w, h) in enumerate(crop_list):
            img_path = os.path.join(images_dir, img_name)
            img = cv2.imread(img_path)
            if img is None: continue
            
            H, W, _ = img.shape
            x_min = int((x - w/2) * W)
            y_min = int((y - h/2) * H)
            x_max = int((x + w/2) * W)
            y_max = int((y + h/2) * H)
            
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(W, x_max), min(H, y_max)
            
            if x_max > x_min and y_max > y_min:
                cropped = img[y_min:y_max, x_min:x_max]
                out_name = f"{img_name.split('.')[0]}_{i}.jpg"
                out_path = os.path.join(output_dir, split_name, class_name_safe, out_name)
                cv2.imwrite(out_path, cropped)

    process_crops(train_crops, "train")
    process_crops(val_crops, "val")
    process_crops(test_crops, "test")

print("Done cropping 80/10/10!")

notebook_path = "paper/AIL_paper.ipynb"
if os.path.exists(notebook_path):
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    if len(nb["cells"]) > 0:
        nb["cells"][0]["source"] = [
            "# Xử lý dataset NVTS\n",
            "1. Loại bỏ các class có < 20 ảnh\n",
            "2. Crop ảnh dựa trên bounding box\n",
            "3. Chia dataset thành 80/10/10 (Train/Val/Test)"
        ]
    
    nb["cells"][1]["source"] = [
        "import os\n",
        "import cv2\n",
        "from collections import defaultdict\n",
        "import random\n",
        "from tqdm import tqdm\n",
        "\n",
        "# Đường dẫn dataset ban đầu\n",
        "dataset_dir = \"dataset_NVTS\"\n",
        "images_dir = os.path.join(dataset_dir, \"images\")\n",
        "labels_dir = os.path.join(dataset_dir, \"labels\")\n",
        "classes_file = os.path.join(dataset_dir, \"classes.txt\")\n",
        "\n",
        "# Thư mục lưu dataset mới\n",
        "output_dir = \"cropped_dataset\"\n",
        "train_dir = os.path.join(output_dir, \"train\")\n",
        "val_dir = os.path.join(output_dir, \"val\")\n",
        "test_dir = os.path.join(output_dir, \"test\")\n",
        "\n",
        "# Đọc danh sách classes\n",
        "with open(classes_file, \"r\") as f:\n",
        "    classes = [line.strip() for line in f.readlines()]\n",
        "\n",
        "print(f\"Tổng số classes ban đầu: {len(classes)}\")"
    ]
    
    nb["cells"][4]["source"] = [
        "# Thực hiện crop và chia 80/10/10\n",
        "os.makedirs(train_dir, exist_ok=True)\n",
        "os.makedirs(val_dir, exist_ok=True)\n",
        "os.makedirs(test_dir, exist_ok=True)\n",
        "\n",
        "for cid, crops in tqdm(class_crops.items(), desc=\"Cropping images\"):\n",
        "    random.seed(42)  # Để đảm bảo chia split giống nhau mỗi lần chạy\n",
        "    random.shuffle(crops)\n",
        "    \n",
        "    n = len(crops)\n",
        "    train_idx = int(n * 0.8)\n",
        "    val_idx = int(n * 0.9)\n",
        "    \n",
        "    train_crops = crops[:train_idx]\n",
        "    val_crops = crops[train_idx:val_idx]\n",
        "    test_crops = crops[val_idx:]\n",
        "    \n",
        "    # Đảm bảo tên thư mục an toàn\n",
        "    class_name_safe = valid_classes_names[cid].replace(\"/\", \"_\").replace(\"*\", \"_\")\n",
        "    os.makedirs(os.path.join(train_dir, class_name_safe), exist_ok=True)\n",
        "    os.makedirs(os.path.join(val_dir, class_name_safe), exist_ok=True)\n",
        "    os.makedirs(os.path.join(test_dir, class_name_safe), exist_ok=True)\n",
        "    \n",
        "    def process_crops(crop_list, split_name):\n",
        "        for i, (img_name, x, y, w, h) in enumerate(crop_list):\n",
        "            img_path = os.path.join(images_dir, img_name)\n",
        "            img = cv2.imread(img_path)\n",
        "            if img is None: continue\n",
        "            \n",
        "            H, W, _ = img.shape\n",
        "            x_min = int((x - w/2) * W)\n",
        "            y_min = int((y - h/2) * H)\n",
        "            x_max = int((x + w/2) * W)\n",
        "            y_max = int((y + h/2) * H)\n",
        "            \n",
        "            # Cắt ảnh theo biên an toàn\n",
        "            x_min, y_min = max(0, x_min), max(0, y_min)\n",
        "            x_max, y_max = min(W, x_max), min(H, y_max)\n",
        "            \n",
        "            if x_max > x_min and y_max > y_min:\n",
        "                cropped = img[y_min:y_max, x_min:x_max]\n",
        "                out_name = f\"{img_name.split('.')[0]}_{i}.jpg\"\n",
        "                out_path = os.path.join(output_dir, split_name, class_name_safe, out_name)\n",
        "                cv2.imwrite(out_path, cropped)\n",
        "\n",
        "    process_crops(train_crops, \"train\")\n",
        "    process_crops(val_crops, \"val\")\n",
        "    process_crops(test_crops, \"test\")\n",
        "\n",
        "print(\"Hoàn tất tạo dataset mới với tập Test!\")"
    ]
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)

