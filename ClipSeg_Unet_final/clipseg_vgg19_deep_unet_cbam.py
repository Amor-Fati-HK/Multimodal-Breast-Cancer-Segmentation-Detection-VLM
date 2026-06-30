
import pandas as pd
import os
import cv2
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
import glob
import warnings
import csv
from collections import OrderedDict
from sklearn.model_selection import train_test_split

# %%
import tensorflow as tf
from tensorflow import keras

# %%
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

# %%
BASE_INPUT_PATH = "/kaggle/input/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset"
CSV_FOLDER_PATH = os.path.join(BASE_INPUT_PATH, "csv")
IMAGE_FOLDER_PATH = os.path.join(BASE_INPUT_PATH, "jpeg")
BASE_OUTPUT_PATH = "/kaggle/working/"

# %%
IMAGE_SIZE = 256
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
RANDOM_SEED = 42

# %%
def find_image_in_folder(folder_path):
    """
    Trouve la première image .jpg ou .png dans le dossier ET ses sous-dossiers.
    """
    if not folder_path or not os.path.isdir(folder_path):
        return None
        
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.png')):
                return os.path.join(root, file)
                
    return None

def compute_all_bounding_boxes(mask_path, min_area=100):
    """
    Returns a list of bounding boxes [[x_min, y_min, width, height],...]
    Returns None if mask doesn't exist or is invalid
    """
    if not os.path.exists(mask_path):
        return None

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    _, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < min_area:
            continue
        boxes.append([x, y, w, h])

    return boxes if boxes else None

def build_metadata_lookup(dicom_info_path, jpeg_base_dir, *args):
    print(f"Building metadata lookup from: {dicom_info_path}")
    master_map = {}
    
    try:
        dicom_info = pd.read_csv(dicom_info_path, dtype=str)
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {dicom_info_path}")
        return master_map
        
    valid_descriptions = {arg for arg in args}
    filtered_df = dicom_info[dicom_info['SeriesDescription'].isin(valid_descriptions)]

    for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Building lookup map"):
        series_desc = row['SeriesDescription'] 
        patient_id_composite = row['PatientID'] 
        
        full_path = None
        
        if 'image_path' in row and pd.notna(row['image_path']):
            rel_path = row['image_path']
            if 'jpeg' in rel_path:
                clean_rel_path = rel_path.split('jpeg')[-1].strip("/\\")
                tmp_path = os.path.join(jpeg_base_dir, clean_rel_path)
            else:
                tmp_path = os.path.join(jpeg_base_dir, rel_path)

            if os.path.exists(tmp_path):
                full_path = tmp_path

        if not full_path:
            series_uid = row['SeriesInstanceUID']
            folder_path = os.path.join(jpeg_base_dir, series_uid)
            full_path = find_image_in_folder(folder_path)
            
        if not full_path or not os.path.exists(full_path): 
            continue

        if patient_id_composite not in master_map:
            master_map[patient_id_composite] = {}
            
        master_map[patient_id_composite][series_desc] = full_path
            
    print(f"Metadata lookup map built. Found {len(master_map)} unique composite keys.")
    return master_map

# %%
def BuildMasterDataset(MASTER_LIST_PATH="/kaggle/working/master_dataset.csv", 
                       argument1="cropped images", 
                       argument2="ROI mask images"):
    BASE_INPUT_PATH = "/kaggle/input/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset"
    IMAGE_FOLDER_PATH = os.path.join(BASE_INPUT_PATH, "jpeg")
    DICOM_INFO_PATH = os.path.join(BASE_INPUT_PATH, "csv/dicom_info.csv")
    
    INPUT_CSVS = [
        os.path.join(BASE_INPUT_PATH, "csv/mass_case_description_train_set.csv"),
        os.path.join(BASE_INPUT_PATH, "csv/mass_case_description_test_set.csv"),
        os.path.join(BASE_INPUT_PATH, "csv/calc_case_description_train_set.csv"),
        os.path.join(BASE_INPUT_PATH, "csv/calc_case_description_test_set.csv")
    ]

    master_map = build_metadata_lookup(DICOM_INFO_PATH, IMAGE_FOLDER_PATH, argument1, argument2)
    
    if not master_map:
        return

    found_pairs_count = 0
    missing_mask_count = 0
    skipped_malignant_count = 0
    
    with open(MASTER_LIST_PATH, 'w', newline='') as outfile:
        csv_writer = csv.writer(outfile)
        csv_writer.writerow([
            'cropped_image_path', 'roi_mask_path',
            'x_min', 'y_min', 'width', 'height',
            'pathology', 'assessment', 'patient_id', 'series_type', 'mask_status',
            'breast_density', 'abnormality_shape', 'abnormality_margin', 'subtlety'
        ])

        for filepath in INPUT_CSVS:
            filename = os.path.basename(filepath)
            if not filepath or not os.path.exists(filepath):
                continue
            
            is_mass = "mass" in filename.lower()
            
            if is_mass:
                type_prefix = "Mass"
            else:
                type_prefix = "Calc"
                
            if "train" in filename.lower():
                split_prefix = "Training"
            else:
                split_prefix = "Test"
                
            full_prefix = f"{type_prefix}-{split_prefix}"

            with open(filepath, "r") as infile:
                csv_reader = csv.reader(infile)
                header = next(csv_reader)
                
                pathology_idx = header.index('pathology')
                assessment_idx = header.index('assessment')
                patient_id_idx = header.index('patient_id')
                breast_idx = header.index('left or right breast')
                view_idx = header.index('image view')
                abnormality_id_idx = header.index('abnormality id')
                
                if 'breast density' in header:
                    density_idx = header.index('breast density')
                else:
                    density_idx = header.index('breast_density')
                
                subtlety_idx = header.index('subtlety')
                
                if is_mass:
                    shape_idx = header.index('mass shape') if 'mass shape' in header else header.index('mass_shape')
                    margin_idx = header.index('mass margins') if 'mass margins' in header else header.index('mass_margins')
                else:
                    shape_idx = header.index('calc type') if 'calc type' in header else header.index('calc_type')
                    margin_idx = header.index('calc distribution') if 'calc distribution' in header else header.index('calc_distribution')

                for row in tqdm(csv_reader, desc=f"Processing {filename}"):
                    if not any(row):
                        continue
                    
                    pathology = row[pathology_idx]
                    assessment = row[assessment_idx]
                    patient_id = row[patient_id_idx]
                    side = row[breast_idx]
                    view = row[view_idx]
                    abn_id = row[abnormality_id_idx]
                    
                    density = row[density_idx]
                    subtlety = row[subtlety_idx]
                    abn_shape = row[shape_idx]
                    abn_margin = row[margin_idx]
                    
                    try:
                        abn_id_clean = str(int(float(abn_id)))
                    except ValueError:
                        abn_id_clean = str(abn_id).strip()

                    composite_key = f"{full_prefix}_{patient_id}_{side}_{view}_{abn_id_clean}"
                    
                    study_data = master_map.get(composite_key)
                    if not study_data:
                        continue

                    full_crop_path = study_data.get('cropped images')
                    full_mask_path = study_data.get('ROI mask images')
                    
                    if not full_crop_path:
                        continue
                    
                    mask_status = 'valid'
                    
                    if not full_mask_path:
                        pathology_upper = str(pathology).upper()
                        is_benign = 'BENIGN' in pathology_upper and 'MALIGNANT' not in pathology_upper
                        
                        if is_benign:
                            full_mask_path = 'n/a'
                            mask_status = 'n/a'
                            missing_mask_count += 1
                            
                            csv_writer.writerow([
                                full_crop_path, 'n/a',
                                'n/a', 'n/a', 'n/a', 'n/a',
                                pathology, assessment, patient_id, full_prefix, 'n/a',
                                density, abn_shape, abn_margin, subtlety
                            ])
                            found_pairs_count += 1
                        else:
                            skipped_malignant_count += 1
                        continue
                    
                    if full_crop_path == full_mask_path and mask_status == 'valid':
                        continue
                    
                    boxes = compute_all_bounding_boxes(full_mask_path, min_area=100)
                     
                    if boxes is None:
                        csv_writer.writerow([
                            full_crop_path, full_mask_path,
                            'n/a', 'n/a', 'n/a', 'n/a',
                            pathology, assessment, patient_id, full_prefix, mask_status,
                            density, abn_shape, abn_margin, subtlety
                        ])
                        found_pairs_count += 1
                    else:
                        for (x_min, y_min, width, height) in boxes:
                            csv_writer.writerow([
                                full_crop_path, full_mask_path,
                                x_min, y_min, width, height,
                                pathology, assessment, patient_id, full_prefix, mask_status,
                                density, abn_shape, abn_margin, subtlety
                            ])
                            found_pairs_count += 1

    print(f"\n{'='*60}")
    print(f"DATASET BUILD SUMMARY")
    print(f"{'='*60}")
    print(f"Master list saved to: {MASTER_LIST_PATH}")
    print(f"Valid pairs found: {found_pairs_count}")
    print(f"Benign cases without masks (n/a): {missing_mask_count}")
    print(f"Malignant cases skipped (no mask): {skipped_malignant_count}")
    print(f"{'='*60}")

BuildMasterDataset()

# %%
MASTER_LIST_PATH = "/kaggle/working/master_dataset.csv"

df_master = pd.read_csv(MASTER_LIST_PATH, keep_default_na=False)

unique_patients = df_master["patient_id"].unique()
train_patients, val_patients = train_test_split(
    unique_patients,
    test_size=VALIDATION_SPLIT,
    random_state=RANDOM_SEED
)

print(f"Splitting {len(unique_patients)} unique patients: {len(train_patients)} for training, {len(val_patients)} for validation.")

# %% [markdown]
# ### Stats

# %%
import pandas as pd
import glob

import seaborn as sns

raw_csv_paths = glob.glob('/kaggle/input/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset/csv/*_case_description_*.csv')
df_raw = pd.concat([pd.read_csv(f) for f in raw_csv_paths])

df_master = pd.read_csv('/kaggle/working/master_dataset.csv')

print("="*50)
print("STATISTIQUES POUR LE RAPPORT")
print("="*50)

print(f"Nombre de lignes initiales (Raw) : {len(df_raw)}")
print(f"Nombre de lignes finales (Master) : {len(df_master)}")
print(f"Différence (Données rejetées) : {len(df_raw) - len(df_master)}")

print("\n" + "="*50)
print("DISTRIBUTION PAR PATHOLOGIE (FINALE)")
print("="*50)
stats_patho = df_master['pathology'].value_counts()
for patho, count in stats_patho.items():
    percentage = (count / len(df_master)) * 100
    print(f"{patho} : {count} images ({percentage:.2f}%)")

print("\n" + "="*50)
print("DISTRIBUTION PAR TYPE (MASS/CALC)")
print("="*50)
stats_type = df_master['series_type'].value_counts()
for s_type, count in stats_type.items():
    print(f"{s_type} : {count} images")


plt.figure(figsize=(10, 5))
sns.countplot(data=df_master, x='pathology', palette='viridis')
plt.title('Distribution des classes dans le Master Dataset')
plt.savefig('distribution_pathologie.png') 
plt.show()

# %%
train_df = df_master[df_master['patient_id'].isin(train_patients)].reset_index(drop=True)
val_df = df_master[df_master['patient_id'].isin(val_patients)].reset_index(drop=True)
train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.Rotate(limit=20, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
    A.GridDistortion(p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
   
])

val_transforms = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    
])

# %%
class MultimodalSequence(tf.keras.utils.Sequence):
    def __init__(self, dataframe, clinical_dict, batch_size=16, img_size=256, transforms=None):
        self.df = dataframe
        self.clinical_dict = clinical_dict
        self.batch_size = batch_size
        self.img_size = img_size
        self.transforms = transforms

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch_df = self.df.iloc[idx * self.batch_size : (idx + 1) * self.batch_size]
        
        imgs, msks, embs = [], [], []

        for _, row in batch_df.iterrows():
            image, mask = self._load_data(row)
            
            p_id = str(row['patient_id'])
            embedding = self.clinical_dict.get(p_id, np.zeros(768))
            
            imgs.append(image)
            msks.append(mask)
            embs.append(embedding)

        return (np.array(imgs), np.array(embs)), np.array(msks)

    def _load_data(self, row):
        
        image_path = row["cropped_image_path"]
        image = cv2.imread(image_path)
        if image is None:
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.img_size, self.img_size))

        mask_path = row["roi_mask_path"]
        if row["mask_status"] == 'n/a' or mask_path == 'n/a':
            mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        else:
            full_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if full_mask is None:
                mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            else:
                # Secure data
                x_min_val = row['x_min']
                y_min_val = row['y_min']
                
                if x_min_val in ['n/a', ''] or pd.isna(x_min_val):
                    x_min = 0
                    y_min = 0
                    x_max = full_mask.shape[1]
                    y_max = full_mask.shape[0]
                else:
                    x_min = max(0, int(float(x_min_val)))
                    y_min = max(0, int(float(y_min_val)))
                    x_max = min(full_mask.shape[1], x_min + int(float(row['width'])))
                    y_max = min(full_mask.shape[0], y_min + int(float(row['height'])))
                
                mask_crop = full_mask[y_min:y_max, x_min:x_max]
                
                if mask_crop.size == 0:
                    mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
                else:
                    mask = cv2.resize(mask_crop, (self.img_size, self.img_size))

        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.float32) / 255.0

        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=-1)
            
        return image, mask

# %%
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from transformers import pipeline

pipe = pipeline("feature-extraction", 
                model="emilyalsentzer/Bio_ClinicalBERT", 
                framework="tf",
                device=-1)

def get_embedding_with_pipe(text):
    features = pipe(text)
    feat_array = np.array(features[0])
    return np.mean(feat_array, axis=0)

print("Reading Master Dataset...")
df_master = pd.read_csv("/kaggle/working/master_dataset.csv", keep_default_na=False)

def combine_features_master(row):
    pathology = str(row['pathology']).replace('_', ' ').lower()
    mask_status = str(row['mask_status']).strip().lower()
    
    if mask_status == 'n/a' or pathology in ['normal', 'nan', '']:
        return "Normal breast tissue, no mass or calcification detected."
    
    shape = str(row['abnormality_shape']).replace('_', ' ').lower()
    margin = str(row['abnormality_margin']).replace('_', ' ').lower()
    
    if shape in ['nan', '']: shape = "irregular"
    if margin in ['nan', '']: margin = "spiculated"
    
    is_mass = "mass" in str(row['series_type']).lower()
    lesion_type = "mass" if is_mass else "calcification"
    
    return f"A {pathology} {lesion_type} with {shape} shape and {margin} margins."

print("Generating training embeddings...")

unique_patients_df = df_master.drop_duplicates(subset=['patient_id'])

clinical_map = {}
for _, row in tqdm(unique_patients_df.iterrows(), total=len(unique_patients_df), desc="Generating Embeddings"):
    text = combine_features_master(row)
    vec = get_embedding_with_pipe(text)
    
    p_id = str(row['patient_id'])
    clinical_map[p_id] = vec

with open("clinical_embeddings_dict_v2.pkl", "wb") as f:
    pickle.dump(clinical_map, f)

print(f"\nDictionary saved! {len(clinical_map)} unique patients processed.")

unique_patients_master = set(df_master["patient_id"].unique())
unique_patients_dict = set(clinical_map.keys())
missing = unique_patients_master - unique_patients_dict

print(f"\n--- VERIFICATION REPORT ---")
print(f"Unique patients in CSV: {len(unique_patients_master)}")
print(f"Patients in dictionary: {len(unique_patients_dict)}")
if len(missing) == 0:
    print(" SUCCESS: 100% of patients are covered by the embedding dictionary!")
else:
    print(f" WARNING: {len(missing)} patients are missing from the dictionary!")

# %%

import pickle
with open("clinical_embeddings_dict_v2.pkl", "rb") as f:
    clinical_dict = pickle.load(f)

train_generator = MultimodalSequence(
    dataframe=train_df, 
    clinical_dict=clinical_dict,
    batch_size=BATCH_SIZE, 
    img_size=IMAGE_SIZE,
    transforms=train_transforms
)

val_generator = MultimodalSequence(
    dataframe=val_df, 
    clinical_dict=clinical_dict,
    batch_size=BATCH_SIZE, 
    img_size=IMAGE_SIZE,
    transforms=val_transforms
)

print(f" Générateurs Multimodaux prêts ")
print(f"Échantillons : Train={len(train_df)} | Val={len(val_df)}")

# %% [markdown]
# ### METRICS

# %%
from tensorflow.keras import backend as K
def dice_coef(y_true,y_pred,smooth=1e-6):
    y_true_f=K.flatten(K.cast(y_true,'float32'))
    y_pred_f=K.flatten(y_pred)

    intersection=K.sum(y_true_f*y_pred_f)
    return (2.*intersection+smooth) / (K.sum(y_true_f)+K.sum(y_pred_f)+smooth)
def dice_loss(y_true,y_pred):
    return 1-dice_coef(y_true,y_pred)

# %%
def specificity(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def f1_score(y_true, y_pred):
    p = keras.metrics.Precision()(y_true, y_pred)
    r = keras.metrics.Recall()(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))

# %% [markdown]
# ### BLOCS AND COMPOSANTS

# %%
def channel_attention_module(x, ratio=8):
    #On recupere le nombre de filtres
    channels=x.shape[-1]
    #On cree les deux neurones MLP
    shared_layer_one=keras.layers.Dense(channels // ratio, activation="relu", use_bias=False)
    shared_layer_two=keras.layers.Dense(channels, use_bias=False)

    # avgpool 
    avg_pool=keras.layers.GlobalAveragePooling2D()(x)
    avg_pool=keras.layers.Reshape((1,1,channels))(avg_pool)
    avg_out=shared_layer_two(shared_layer_one(avg_pool))

    #maxpool
    max_pool=keras.layers.GlobalMaxPooling2D()(x)
    max_pool=keras.layers.Reshape((1,1,channels))(max_pool)
    max_out=shared_layer_two(shared_layer_one(max_pool))

    #Addition et sigmoide
    cbam_feature=keras.layers.Add()([avg_out,max_out])
    cbam_feature=keras.layers.Activation('sigmoid')(cbam_feature)

    #Multiplication 
    return keras.layers.multiply([x,cbam_feature])

# %%
from keras import ops
def spatial_attention_module(x):
    #
    avg_pool = ops.mean(x,axis=-1,keepdims=True)
    max_pool= ops.max(x,axis=-1,keepdims=True)

    #concatenation
    concat=keras.layers.Concatenate(axis=-1)([avg_pool, max_pool])

    #7x7 filter et sigmoide
    cbam_feature=keras.layers.Conv2D(
        filters=1, kernel_size=7, strides=1,
        padding="same", activation="sigmoid", use_bias=False
    )(concat)

    #muliply 
    return keras.layers.multiply([x,cbam_feature])

# %%
def cbam_block(x,ratio=8):
    x=channel_attention_module(x,ratio)
    x=spatial_attention_module(x)
    return x

# %%
def cross_attention_gate(img_features, bert_vector, channels):
    """
    Vraie Cross-Attention (Query-Key-Value)
    Image = Query | Texte = Key & Value
    """
    bert_quiet = keras.layers.Dropout(0.8)(bert_vector)
    bert_norm = keras.layers.BatchNormalization()(bert_quiet)

    query = keras.layers.Conv2D(channels, 1, padding='same')(img_features)

    key = keras.layers.Dense(channels)(bert_norm)
    key = keras.layers.Reshape((1, 1, channels))(key)
    
    value = keras.layers.Dense(channels)(bert_norm)
    value = keras.layers.Reshape((1, 1, channels))(value)

    attention_map = keras.layers.Multiply()([query, key])
    

    attention_map = keras.layers.Activation('sigmoid')(attention_map) 

    out = keras.layers.Multiply()([attention_map, value])
    
    return keras.layers.Add()([img_features, out])

# %%
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class DDSM_CLIPSeg_Dataset(Dataset):
    def __init__(self, dataframe, processor):
        self.df = dataframe[dataframe['mask_status'] != 'n/a'].reset_index(drop=True)
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['cropped_image_path']).convert("RGB")
        full_mask = Image.open(row['roi_mask_path']).convert("L")
        
        x_min, y_min = max(0, int(float(row['x_min']))), max(0, int(float(row['y_min'])))
        x_max = min(full_mask.size[0], x_min + int(float(row['width'])))
        y_max = min(full_mask.size[1], y_min + int(float(row['height'])))
        mask_crop = full_mask.crop((x_min, y_min, x_max, y_max))

        pathology = str(row['pathology']).replace('_', ' ').lower()
        assessment = str(row['assessment'])
        prompt = f"a {pathology} mass with BI-RADS assessment {assessment}"
        
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt", padding="max_length")
        
        mask_res = mask_crop.resize((352, 352))
        mask_tensor = torch.tensor(np.array(mask_res)).float() / 255.0
        
        return {
            "pixel_values": inputs.pixel_values.squeeze(0),
            "input_ids": inputs.input_ids.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0),
            "labels": mask_tensor
        }

# %%
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
from torch.optim import Adam

processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

for param in model.clip.parameters():
    param.requires_grad = False

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
optimizer = Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.BCEWithLogitsLoss()

train_ds = DDSM_CLIPSeg_Dataset(train_df, processor)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)

print("Fine-tuning de CLIPSeg ...")
model.train()
for epoch in range(3):
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device).unsqueeze(1) # [B, 1, 352, 352]

        outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        if len(logits.shape) == 3:
            logits = logits.unsqueeze(1) 
            
        logits_resized = torch.nn.functional.interpolate(logits, size=(352, 352), mode="bilinear", align_corners=False)
        
        loss = criterion(logits_resized, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "clipseg_ddsm_fine_tuned.pth")

# %%
def dice_coef_pytorch(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.view(-1)
    y_pred_f = torch.sigmoid(y_pred).view(-1)
    intersection = (y_true_f * y_pred_f).sum()
    return (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)

# Final Config
val_ds = DDSM_CLIPSeg_Dataset(val_df, processor)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)
optimizer = Adam(model.parameters(), lr=5e-5) # LR a bit lower 

best_dice = 0

print(" Lancement du Fit Final ClipSeg...")
for epoch in range(10): # 10 epochs are enough
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/10"):
        optimizer.zero_grad()
        logits = model(pixel_values=batch["pixel_values"].to(device),
                       input_ids=batch["input_ids"].to(device),
                       attention_mask=batch["attention_mask"].to(device)).logits
        logits = torch.nn.functional.interpolate(logits.unsqueeze(1), size=(352, 352), mode="bilinear")
        loss = criterion(logits, batch["labels"].to(device).unsqueeze(1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_dice = 0
    with torch.no_grad():
        for batch in val_loader:
            logits = model(pixel_values=batch["pixel_values"].to(device),
                           input_ids=batch["input_ids"].to(device),
                           attention_mask=batch["attention_mask"].to(device)).logits
            logits = torch.nn.functional.interpolate(logits.unsqueeze(1), size=(352, 352), mode="bilinear")
            val_dice += dice_coef_pytorch(batch["labels"].to(device), logits)
    
    avg_dice = val_dice / len(val_loader)
    print(f" Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | Val Dice: {avg_dice:.4f}")
    
    if avg_dice > best_dice:
        best_dice = avg_dice
        torch.save(model.state_dict(), "best_clipseg_model.pth")
        print(" Nouveau modèle sauvegardé ")

# %% [markdown]
# ### Model Architecture 

# %%
import torch.nn as nn
class CBAM(nn.Module):
    def __init__(self, channels):
        super(CBAM, self).__init__()
        # Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // 8, channels, 1, bias=False),
            nn.Sigmoid()
        )
        # Spatial Attention
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x * self.ca(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = self.sa(torch.cat([avg_out, max_out], dim=1))
        return x * spatial


import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
class VGG19_CLIP_Hybrid(nn.Module):
    def __init__(self, frozen_clipseg):
        super(VGG19_CLIP_Hybrid_SOTA, self).__init__()
        self.clipseg = frozen_clipseg
        for param in self.clipseg.parameters(): param.requires_grad = False
        
        # Vgg19 encoder
        vgg = models.vgg19(pretrained=True).features
        self.enc1, self.enc2 = vgg[:4], vgg[4:9]
        self.enc3, self.enc4 = vgg[9:18], vgg[18:27]
        self.enc5 = vgg[27:36]
        
        # Fine tuning
        for p in self.enc1.parameters(): p.requires_grad = False
        for p in self.enc2.parameters(): p.requires_grad = False

        # Attention (CBAM)
        self.cbam5 = CBAM(512)
        self.cbam4 = CBAM(512)
        self.cbam3 = CBAM(256)
        self.cbam2 = CBAM(128)
        self.cbam1 = CBAM(64)

        #Decoder
        self.up4 = nn.ConvTranspose2d(512 + 1, 512, 2, 2) 
        self.dec4 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), # Double Conv !
            nn.Dropout(0.2)
        )
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU()
        )
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), 
                                  nn.Conv2d(64, 1, 1), nn.Sigmoid())

    def forward(self, pixel_values, input_ids, attention_mask):
        with torch.no_grad():
            logits = self.clipseg(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask).logits
            heatmap = torch.sigmoid(logits).unsqueeze(1)

        s1 = self.cbam1(self.enc1(pixel_values)) 
        s2 = self.cbam2(self.enc2(s1))    
        s3 = self.cbam3(self.enc3(s2))    
        s4 = self.cbam4(self.enc4(s3))    
        b  = self.cbam5(self.enc5(s4))     

        h_b = F.interpolate(heatmap, size=(b.shape[2], b.shape[3]), mode="bilinear", align_corners=True)
        b_fused = b * h_b 
        
        x = self.up4(b_fused) 
        x = torch.cat([x, s4], dim=1); x = self.dec4(x)
        x = self.up3(x); x = torch.cat([x, s3], dim=1); x = self.dec3(x)
        x = self.up2(x); x = torch.cat([x, s2], dim=1); x = self.dec2(x)
        x = self.up1(x); x = torch.cat([x, s1], dim=1); out = self.dec1(x)

        return out

model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
PATH_CLIP_BEST = "/kaggle/input/datasets/hanikatti/mes-modles-pfe/best_clipseg_model(2).pth"
model.load_state_dict(torch.load(PATH_CLIP_BEST, map_location=device))
model.eval()
hybrid_model_v1 = VGG19_CLIP_Hybrid(model).to(device)

# %% [markdown]
# ### Without Fine Tuning

# %%
from torch.optim import Adam
from tqdm import tqdm

print("Fitting the model...")


for param in hybrid_model_v1.enc1.parameters(): param.requires_grad = False
for param in hybrid_model_v1.enc2.parameters(): param.requires_grad = False
for param in hybrid_model_v1.enc3.parameters(): param.requires_grad = False
for param in hybrid_model_v1.enc4.parameters(): param.requires_grad = False
for param in hybrid_model_v1.enc5.parameters(): param.requires_grad = False

optimizer_hybrid = Adam(filter(lambda p: p.requires_grad, hybrid_model_v1.parameters()), lr=1e-4)
criterion = torch.nn.BCELoss()

for epoch in range(5): 
    
    hybrid_model_v1.train()
    
    total_loss = 0
    for batch in tqdm(train_loader):
        optimizer_hybrid.zero_grad()
        
        pred_mask = hybrid_model_v1(
            pixel_values=batch["pixel_values"].to(device),
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device)
        )
        
        labels = batch["labels"].to(device).unsqueeze(1)
        loss = criterion(pred_mask, labels) 
        
        loss.backward()
        optimizer_hybrid.step()
        
        total_loss += loss.item()
        
    print(f" Epoch {epoch+1} ended | Loss: {total_loss/len(train_loader):.4f}")

# %% [markdown]
# ### First look before fine tune

# %%
def visualize_hybrid_results(idx):
    hybrid_model_v1.eval() 
    row = val_df.iloc[idx]
    
    image_raw = Image.open(row['cropped_image_path']).convert("RGB")
    pathology = str(row['pathology']).replace('_', ' ').lower()
    prompt = f"a {pathology} mass"
    
    full_mask = Image.open(row['roi_mask_path']).convert("L")
    x_min, y_min = max(0, int(float(row['x_min']))), max(0, int(float(row['y_min'])))
    x_max = min(full_mask.size[0], x_min + int(float(row['width'])))
    y_max = min(full_mask.size[1], y_min + int(float(row['height'])))
    true_mask = full_mask.crop((x_min, y_min, x_max, y_max))
    true_mask = np.array(true_mask.resize((image_raw.size[0], image_raw.size[1])))

    inputs = processor(text=[prompt], images=[image_raw], return_tensors="pt", padding="max_length").to(device)
    
    with torch.no_grad():
        clip_outputs = hybrid_model_v1.clipseg(**inputs)
        heatmap = torch.sigmoid(clip_outputs.logits).unsqueeze(1)
        heatmap_inter = torch.nn.functional.interpolate(heatmap, size=(352, 352), mode="bilinear")
        final_mask_tensor = hybrid_model_v1(inputs.pixel_values, inputs.input_ids, inputs.attention_mask)

    heatmap_img = heatmap_inter.squeeze().cpu().numpy()
    heatmap_img = cv2.resize(heatmap_img, (image_raw.size[0], image_raw.size[1]))
    
    final_mask_img = final_mask_tensor.squeeze().cpu().numpy()
    final_mask_img = cv2.resize(final_mask_img, (image_raw.size[0], image_raw.size[1]))

    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(image_raw)
    plt.title(f"1. Original Image\n({pathology})")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(true_mask, cmap="gray")
    plt.title("2. Ground Truth")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(heatmap_img, cmap="jet")
    plt.title("3. Heatmap CLIPSeg")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(final_mask_img > 0.3, cmap="gray")
    plt.title("4. Prediction (AI)")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

for i in [4, 39, 10]:
    visualize_hybrid_results(i)

# %% [markdown]
# ### Fine Tuning VGG19

# %%

from torch.optim import Adam
from tqdm import tqdm

print("Fitting the model...")


for param in hybrid_model_v1.enc4.parameters(): param.requires_grad = True
for param in hybrid_model_v1.enc5.parameters(): param.requires_grad = True
    
optimizer_hybrid = Adam(filter(lambda p: p.requires_grad, hybrid_model_v1.parameters()), lr=1e-5)
criterion = torch.nn.BCELoss() 

for epoch in range(10): 
    
    hybrid_model_v1.train()
    
    total_loss = 0
    for batch in tqdm(train_loader):
        optimizer_hybrid.zero_grad()
        
        pred_mask = hybrid_model_v1(
            pixel_values=batch["pixel_values"].to(device),
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device)
        )
        
        labels = batch["labels"].to(device).unsqueeze(1)
        loss = criterion(pred_mask, labels) 
        
        loss.backward()
        optimizer_hybrid.step()
        
        total_loss += loss.item()
        
    print(f" Epoch {epoch+1} ended | Loss: {total_loss/len(train_loader):.4f}")

# %% [markdown]
# ### Final results

# %%
def visualize_hybrid_results(idx):
    hybrid_model_v1.eval() 
    row = val_df.iloc[idx]
    
    image_raw = Image.open(row['cropped_image_path']).convert("RGB")
    pathology = str(row['pathology']).replace('_', ' ').lower()
    prompt = f"a {pathology} mass"
    
    full_mask = Image.open(row['roi_mask_path']).convert("L")
    x_min, y_min = max(0, int(float(row['x_min']))), max(0, int(float(row['y_min'])))
    x_max = min(full_mask.size[0], x_min + int(float(row['width'])))
    y_max = min(full_mask.size[1], y_min + int(float(row['height'])))
    true_mask = full_mask.crop((x_min, y_min, x_max, y_max))
    true_mask = np.array(true_mask.resize((image_raw.size[0], image_raw.size[1])))

    inputs = processor(text=[prompt], images=[image_raw], return_tensors="pt", padding="max_length").to(device)
    
    with torch.no_grad():
        clip_outputs = hybrid_model_v1.clipseg(**inputs)
        heatmap = torch.sigmoid(clip_outputs.logits).unsqueeze(1)
        heatmap_inter = torch.nn.functional.interpolate(heatmap, size=(352, 352), mode="bilinear")
        final_mask_tensor = hybrid_model_v1(inputs.pixel_values, inputs.input_ids, inputs.attention_mask)

    heatmap_img = heatmap_inter.squeeze().cpu().numpy()
    heatmap_img = cv2.resize(heatmap_img, (image_raw.size[0], image_raw.size[1]))
    
    final_mask_img = final_mask_tensor.squeeze().cpu().numpy()
    final_mask_img = cv2.resize(final_mask_img, (image_raw.size[0], image_raw.size[1]))

    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(image_raw)
    plt.title(f"1. Original Image\n({pathology})")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(true_mask, cmap="gray")
    plt.title("2. Ground Truth")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(heatmap_img, cmap="jet")
    plt.title("3. Heatmap CLIPSeg")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(final_mask_img > 0.3, cmap="gray")
    plt.title("4. Prediction (AI)")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

for i in [4, 39, 10]:
    visualize_hybrid_results(i)

# %%
def evaluate_hybrid_complete(model, loader):
    model.eval()
    tp_total, fp_total, fn_total, tn_total = 0, 0, 0, 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Clinical Evaluation"):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device).unsqueeze(1)

            outputs = model(pixel_values, input_ids, attention_mask)
            preds = (outputs > 0.5).float() 
            
            tp_total += (preds * labels).sum().item()
            fp_total += (preds * (1 - labels)).sum().item()
            fn_total += ((1 - preds) * labels).sum().item()
            tn_total += ((1 - preds) * (1 - labels)).sum().item()

    epsilon = 1e-7
    accuracy = (tp_total + tn_total) / (tp_total + tn_total + fp_total + fn_total + epsilon)
    precision = tp_total / (tp_total + fp_total + epsilon)
    recall = tp_total / (tp_total + fn_total + epsilon)
    specificity = tn_total / (tn_total + fp_total + epsilon)

    dice = (2 * tp_total) / (2 * tp_total + fp_total + fn_total + epsilon)
    f1_score = dice 

    results = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall (Sensitivity)": recall,
        "Specificity": specificity,
        "Dice Coefficient": dice,
        "F1-Score": f1_score
    }

    print("\nTABLEAU DES MÉTRIQUES CLINIQUES :")
    print("-" * 40)
    for k, v in results.items():
        print(f"{k:<25} : {v:.4f}")
    print("-" * 40)
    
    return results

stats_finales = evaluate_hybrid_complete(hybrid_model_v1, val_loader)


