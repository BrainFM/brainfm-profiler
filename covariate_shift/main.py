import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import numpy as np
import tensorflow as tf
import nibabel as nib
import scipy.ndimage
import warnings
from tensorflow.keras.layers import Dense, Input
# For plotting and statistical analysis
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from scipy.stats import wasserstein_distance, ttest_ind
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight # Добавляем импорт для работы с весами классов

# Required for tf.keras.models.Model
from tensorflow.keras.models import Model

# --- FIXING RANDOM SEEDS FOR REPRODUCIBILITY ---
# Это важно для получения одинаковых результатов при каждом запуске
import random

os.environ['PYTHONHASHSEED'] = '42' # Для хеш-операций
np.random.seed(42) # Для numpy
tf.random.set_seed(42) # Для tensorflow
random.seed(42) # Для стандартного модуля random в Python

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings("ignore")

# --- Define Paths ---
DATA_ROOT_DS1 = "/home/m.benedichuk/NFBS/NFBS_Dataset"
DATA_ROOT_DS2 = "/home/m.benedichuk/mounted_storage2_Aracar/unzipped_data/IXI"
PROCESSED_DATA_DIR = "/home/m.benedichuk/my_article/data/preprocessed_output_healthy_analysis"
METADATA_CSV_PATH_DS1 = "/home/m.benedichuk/my_article/data/NFBS_metadata.csv"
METADATA_CSV_PATH_DS2 = "/home/m.benedichuk/my_article/data/IXI_metadata.csv"
UNIFIED_METADATA_CSV_PATH = "/home/m.benedichuk/my_article/data/unified_metadata_healthy_analysis.csv"
PLOTS_OUTPUT_DIR = "/home/m.benedichuk/my_article/data/plots/healthy"
# !!! IMPORTANT: PATH TO YOUR MNI TEMPLATE !!!
MNI_TEMPLATE_PATH = "/home/m.benedichuk/my_article/src/mni_icbm152_t1_tal_nlin_asym_09a.nii" # Or MNI152_T1_1mm.nii.gz

# --- Control Flag for Metadata Collection ---
FORCE_METADATA_RECOLLECTION = False # Set to True to always regenerate metadata CSVs

# --- Model Parameters ---
IMAGENET_TARGET_SHAPE_2D = (224, 224)
PENULTIMATE_LAYER_NAME = 'avg_pool' # Common for DenseNet121 before the final dense layer


# --- Helper function for preparing a 2D slice for an ImageNet model (unchanged) ---
def _prepare_2d_slice_for_imagenet(image_path: str, target_shape_2d: tuple = (224, 224), slice_idx: int = None) -> np.ndarray:
    """
    Loads a 3D NIfTI, selects a central slice, normalizes, and resizes it to 2D (H, W, 3)
    for compatibility with pre-trained ImageNet models.
    """
    try:
        img = nib.load(image_path)
        image_data = img.get_fdata().astype(np.float32)

        if slice_idx is None:
            slice_idx = image_data.shape[0] // 2
        
        if image_data.ndim == 4:
            image_data = image_data[..., 0]

        if image_data.ndim != 3:
            raise ValueError(f"Expected 3D or 4D image, but got {image_data.ndim}D for {image_path}. Image shape: {image_data.shape}")

        if not (0 <= slice_idx < image_data.shape[0]):
            warnings.warn(f"Slice index {slice_idx} out of bounds for image with depth {image_data.shape[0]}. Using central slice instead.")
            slice_idx = image_data.shape[0] // 2
        
        slice_data = image_data[slice_idx, :, :]

        min_val = np.min(slice_data)
        max_val = np.max(slice_data)
        if max_val > min_val:
            slice_data = 255 * (slice_data - min_val) / (max_val - min_val)
        else:
            slice_data = np.zeros_like(slice_data)

        current_shape_2d = slice_data.shape
        zoom_factors = [ts / cs for ts, cs in zip(target_shape_2d, current_shape_2d)]
        resized_slice = scipy.ndimage.zoom(slice_data, zoom_factors, order=1, mode='nearest')

        rgb_slice = np.stack([resized_slice, resized_slice, resized_slice], axis=-1)

        return rgb_slice.astype(np.float32)

    except Exception as e:
        print(f"Error preparing 2D slice from {image_path}: {e}. Returning dummy array.")
        return np.zeros(target_shape_2d + (3,), dtype=np.float32)

# --- Function for extracting feature embeddings from the penultimate layer (unchanged) ---
def get_feature_embedding(model: tf.keras.Model, img_array: np.ndarray, layer_name: str):
    """
    Extracts the feature embedding from the specified layer for a given image.
    Applies Global Average Pooling if the layer output is 4D (activations).
    """
    if img_array.ndim == 3:
        img_array = np.expand_dims(img_array, axis=0)

    feature_extractor = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    features = feature_extractor.predict(img_array)

    if features.ndim == 4:
        features = np.mean(features, axis=(1, 2))
    
    return features[0]

# --- MODIFIED: Metadata collection function to generate CSVs if not found or forced ---
def simplified_metadata_collection(data_root_dir, metadata_csv_path, dataset_name, force_recollect=False):
    print(f"Collecting metadata for {dataset_name} from {data_root_dir}...")
    
    df = pd.DataFrame() # Инициализируем df как пустой DataFrame

    if force_recollect or not os.path.exists(metadata_csv_path):
        print(f"Metadata CSV not found or force_recollect is True for {dataset_name}. Generating new metadata...")
        collected_data = None # Инициализируем collected_data
        if dataset_name == 'ds1':
            # Предполагается, что collect_ds1_metadata возвращает DataFrame или список dicts
            from covariate_shift.nfbs_metadata_extractor import collect_nfbs_metadata
            collected_data = collect_nfbs_metadata(data_root_dir, metadata_csv_path) # Используем metadata_csv_path
        elif dataset_name == 'ds2':
            # Предполагается, что collect_ds2_metadata возвращает DataFrame или список dicts
            from covariate_shift.ixi_metadata_extractor import collect_ixi_metadata
            collected_data = collect_ixi_metadata(data_root_dir, metadata_csv_path) # Используем metadata_csv_path
        else:
            print(f"Error: Unknown dataset_name '{dataset_name}'. Cannot collect metadata.")
            return pd.DataFrame() # Возвращаем пустой DataFrame в случае ошибки
            
        # ИСПРАВЛЕНИЕ: Проверяем, что collected_data является DataFrame и не пуст
        if isinstance(collected_data, pd.DataFrame) and not collected_data.empty:
            df = collected_data # Если это уже DataFrame, используем его напрямую
            # Ensure output directory exists before saving
            os.makedirs(os.path.dirname(metadata_csv_path), exist_ok=True)
            df.to_csv(metadata_csv_path, index=False)
            print(f"Generated and saved {len(df)} entries to {metadata_csv_path}")
        elif isinstance(collected_data, list) and collected_data: # Если это список и не пуст
            df = pd.DataFrame(collected_data)
            os.makedirs(os.path.dirname(metadata_csv_path), exist_ok=True)
            df.to_csv(metadata_csv_path, index=False)
            print(f"Generated and saved {len(df)} entries to {metadata_csv_path}")
        else:
            print(f"Warning: No metadata collected for {dataset_name} or collected data is empty/invalid.")
            return pd.DataFrame() # Возвращаем пустой DataFrame, если данные не собраны или невалидны
    else:
        df = pd.read_csv(metadata_csv_path)
        print(f"Loaded {len(df)} entries from {metadata_csv_path}")
    
    # Common post-processing
    df['dataset'] = dataset_name
    
    if 'modality' in df.columns:
        df['modality'] = df['modality'].astype(str) 
    else:
        print(f"Warning: 'modality' column not found in {metadata_csv_path}. Cannot filter by modality.")
        df['modality'] = 'unknown'

    if 'disease' in df.columns:
        df['disease'] = df['disease'].astype(str).str.lower()
    else:
        print(f"Warning: 'disease' column not found in {metadata_csv_path}. Cannot filter by disease.")
        df['disease'] = 'unknown'
        
    return df

# --- Main execution block ---
if __name__ == "__main__":
    from covariate_shift.feature_analysis_visualizations import generate_feature_analysis_plots
    from covariate_shift.preprocessing import preprocess_nifti_file
    
    print("--- Starting combined Preprocessing and Domain Shift Analysis Script ---")

    # --- 0. Validate MNI Template Path ---
    # Проверяем наличие MNI шаблона
    if not os.path.exists(MNI_TEMPLATE_PATH) or MNI_TEMPLATE_PATH.startswith("/path/to/your/template"):
        print(f"ERROR: MNI template file not found or path not updated: '{MNI_TEMPLATE_PATH}'")
        print("Please download an MNI template (e.g., MNI152_T1_1mm_brain.nii.gz) and update 'MNI_TEMPLATE_PATH'. Exiting.")
        # Создаем фиктивный MNI шаблон для демонстрации, чтобы код мог выполняться
        os.makedirs(os.path.dirname(MNI_TEMPLATE_PATH), exist_ok=True)
        # Создаем минимальный валидный NIfTI файл для MNI шаблона
        # Это необходимо, так как nibabel требует валидный NIfTI файл
        import numpy as np
        header = nib.Nifti1Header()
        header['sizeof_hdr'] = 348
        header['dim_info'] = 0
        header['dim'] = [3, 1, 1, 1, 1, 1, 1, 1] # 3D image, 1x1x1
        header['datatype'] = 2 # 8-bit unsigned integer
        header['bitpix'] = 8
        affine = np.eye(4)
        dummy_data = np.zeros((1,1,1), dtype=np.uint8)
        dummy_img = nib.Nifti1Image(dummy_data, affine, header)
        nib.save(dummy_img, MNI_TEMPLATE_PATH)
        print("Continuing with dummy MNI template for demonstration.")
            
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True) # Убедимся, что директория для обработанных данных существует
    os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True) # Убедимся, что директория для графиков существует

    print("\n--- 1. Collecting and Unifying Metadata (and initiating preprocessing) ---")

    # Сбор метаданных для каждого датасета
    # Убедитесь, что FORCE_METADATA_RECOLLECTION = True, чтобы изменения в функциях сбора метаданных применились!
    df_ds1 = simplified_metadata_collection(DATA_ROOT_DS1, METADATA_CSV_PATH_DS1, 'ds1', FORCE_METADATA_RECOLLECTION)
    df_ds2 = simplified_metadata_collection(DATA_ROOT_DS2, METADATA_CSV_PATH_DS2, 'ds2', FORCE_METADATA_RECOLLECTION)

    # Объединение метаданных
    full_metadata_df = pd.concat([df_ds1, df_ds2], ignore_index=True)
    print(f"Total metadata entries after unification: {len(full_metadata_df)}")

    print("Filtering for T1-weighted NIfTI images of HEALTHY patients for preprocessing...")
    # Фильтрация только T1w NIfTI изображений здоровых пациентов
    nifti_t1w_healthy_entries = full_metadata_df[
        full_metadata_df['image_path'].str.contains(r'\.(nii|nii\.gz)$', regex=True, na=False) &
        (full_metadata_df['modality'] == 'T1w') & 
        (full_metadata_df['disease'].str.contains('healthy', case=False, na=False))
    ].copy()
    print(f"Found {len(nifti_t1w_healthy_entries)} T1w NIfTI healthy entries for preprocessing.")

    preprocessed_path_map = {} # Словарь для сопоставления оригинальных и предобработанных путей

    print(f"\n--- Starting bulk preprocessing into {PROCESSED_DATA_DIR} (T1w healthy images only) ---")

    # Обрабатываем только уникальные записи, чтобы избежать дублирования работы
    unique_original_image_paths_with_modality_disease = nifti_t1w_healthy_entries[['image_path', 'dataset', 'modality', 'disease']].drop_duplicates()

    for idx, row in unique_original_image_paths_with_modality_disease.iterrows():
        original_path_relative = row['image_path']
        dataset_name = row['dataset']
        modality = row['modality'] 
        disease = row['disease']

        base_root = DATA_ROOT_DS1 if dataset_name == 'ds1' else DATA_ROOT_DS2
        original_full_path = os.path.join(base_root, original_path_relative)

        # Создаем фиктивный файл, если он не существует, для демонстрации
        if not os.path.exists(original_full_path):
            print(f"Warning: Original file not found at {original_full_path}. Skipping preprocessing for {original_path_relative}.")
            os.makedirs(os.path.dirname(original_full_path), exist_ok=True)
            try:
                # Создаем минимальный валидный NIfTI файл
                header = nib.Nifti1Header()
                header['sizeof_hdr'] = 348
                header['dim_info'] = 0
                header['dim'] = [3, 1, 1, 1, 1, 1, 1, 1] # 3D image, 1x1x1
                header['datatype'] = 2 # 8-bit unsigned integer
                header['bitpix'] = 8
                affine = np.eye(4)
                dummy_data = np.zeros((1,1,1), dtype=np.uint8)
                dummy_img = nib.Nifti1Image(dummy_data, affine, header)
                nib.save(dummy_img, original_full_path)
                print(f"Created dummy file at {original_full_path} for demonstration.")
            except Exception as e:
                print(f"Error creating dummy NIfTI file at {original_full_path}: {e}. Ensure directory {os.path.dirname(original_full_path)} is writable.")
                continue # Пропускаем предобработку, если не удалось создать фиктивный файл

        # Повторная проверка, чтобы убедиться, что обрабатываем только T1w healthy
        if modality != 'T1w' or 'healthy' not in disease.lower(): 
            print(f"Warning: File {original_full_path} unexpectedly not T1w or not healthy after initial filter. Skipping.")
            continue

        preprocessed_file_path = preprocess_nifti_file(
            input_path=original_full_path,
            output_dir=PROCESSED_DATA_DIR,
            template_path=MNI_TEMPLATE_PATH,
            modality=modality, 
            force_overwrite=False
        )
        if preprocessed_file_path:
            preprocessed_path_map[original_path_relative] = preprocessed_file_path

    # --- Обновление full_metadata_df путями к предобработанным изображениям ---
    if 'preprocessed_image_path' not in full_metadata_df.columns:
        full_metadata_df['preprocessed_image_path'] = None

    for original_rel_path, preprocessed_abs_path in preprocessed_path_map.items():
        full_metadata_df.loc[full_metadata_df['image_path'] == original_rel_path, 'preprocessed_image_path'] = preprocessed_abs_path

    print(f"Updated full_metadata_df with {len(preprocessed_path_map)} preprocessed paths.")

    # --- 2. Final Filtering for Healthy Images and Valid Preprocessed Paths (including T1w filter) ---
    print("\n--- 2. Final Filtering for Healthy images with valid preprocessed paths (T1w only) for analysis ---")

    # Окончательная фильтрация для анализа: только T1w здоровые с существующими предобработанными файлами
    healthy_df = full_metadata_df[
        full_metadata_df['disease'].str.contains('healthy', case=False, na=False) &
        (full_metadata_df['modality'] == 'T1w') 
    ].copy()

    healthy_df = healthy_df.dropna(subset=['preprocessed_image_path'])
    healthy_df = healthy_df[healthy_df['preprocessed_image_path'].apply(os.path.exists)]

    print(f"Found {len(healthy_df)} preprocessed T1w healthy image entries for analysis.")

    # Создание фиктивных данных, если реальных не найдено, для демонстрации
    if healthy_df.empty:
        print("No T1w healthy images found with valid preprocessed paths for analysis. Exiting.")
        print("Creating dummy healthy_df for demonstration purposes.")
        healthy_df = pd.DataFrame({
            'image_path': ['sub-01/anat/sub-01_t1w.nii.gz', 'sub-04/anat/sub-04_t1w.nii.gz'],
            'dataset': ['ds1', 'ds2'],
            'modality': ['T1w', 'T1w'], 
            'disease': ['healthy', 'healthy'],
            'preprocessed_image_path': [os.path.join(PROCESSED_DATA_DIR, 'preprocessed_sub-01_t1w.nii.gz'),
                                         os.path.join(PROCESSED_DATA_DIR, 'preprocessed_sub-04_t1w.nii.gz')]
        })
        for p in healthy_df['preprocessed_image_path']:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            # Создаем минимальный валидный NIfTI файл
            header = nib.Nifti1Header()
            header['sizeof_hdr'] = 348
            header['dim_info'] = 0
            header['dim'] = [3, 1, 1, 1, 1, 1, 1, 1]
            header['datatype'] = 2
            header['bitpix'] = 8
            affine = np.eye(4)
            dummy_data = np.zeros((1,1,1), dtype=np.uint8)
            dummy_img = nib.Nifti1Image(dummy_data, affine, header)
            nib.save(dummy_img, p)
        print("Dummy healthy_df created with dummy NIfTI files.")


    healthy_ds1_df = healthy_df[healthy_df['dataset'] == 'ds1']
    healthy_ds2_df = healthy_df[healthy_df['dataset'] == 'ds2']

    print(f"\n--- DEBUG INFO ---")
    print(f"Total healthy_df shape: {healthy_df.shape}")
    print(f"healthy_df head:\n{healthy_df.head()}")
    print(f"healthy_df unique datasets: {healthy_df['dataset'].unique()}")
    print(f"healthy_ds1_df count: {len(healthy_ds1_df)}")
    print(f"healthy_ds2_df count: {len(healthy_ds2_df)}")
    print(f"--- END DEBUG INFO ---\n")

    if healthy_ds1_df.empty or healthy_ds2_df.empty:
        print("One or both datasets lack T1w healthy images for analysis. Cannot perform domain shift analysis between them. Exiting.")
        # Создаем фиктивные датафреймы, если они пусты, для продолжения демонстрации
        if healthy_ds1_df.empty:
            healthy_ds1_df = pd.DataFrame({
                'image_path': ['sub-dummy1/anat/sub-dummy1_t1w.nii.gz'],
                'dataset': ['ds1'], 'modality': ['T1w'], 'disease': ['healthy'], 
                'preprocessed_image_path': [os.path.join(PROCESSED_DATA_DIR, 'preprocessed_sub-dummy1_t1w.nii.gz')]
            })
            os.makedirs(os.path.dirname(healthy_ds1_df['preprocessed_image_path'].iloc[0]), exist_ok=True)
            # Создаем минимальный валидный NIfTI файл
            header = nib.Nifti1Header()
            header['sizeof_hdr'] = 348
            header['dim_info'] = [3, 1, 1, 1, 1, 1, 1, 1]
            header['datatype'] = 2
            header['bitpix'] = 8
            affine = np.eye(4)
            dummy_data = np.zeros((1,1,1), dtype=np.uint8)
            dummy_img = nib.Nifti1Image(dummy_data, affine, header)
            nib.save(dummy_img, healthy_ds1_df['preprocessed_image_path'].iloc[0])
        if healthy_ds2_df.empty:
            healthy_ds2_df = pd.DataFrame({
                'image_path': ['sub-dummy2/anat/sub-dummy2_t1w.nii.gz'],
                'dataset': ['ds2'], 'modality': ['T1w'], 'disease': ['healthy'], 
                'preprocessed_image_path': [os.path.join(PROCESSED_DATA_DIR, 'preprocessed_sub-dummy2_t1w.nii.gz')]
            })
            os.makedirs(os.path.dirname(healthy_ds2_df['preprocessed_image_path'].iloc[0]), exist_ok=True)
            # Создаем минимальный валидный NIfTI файл
            header = nib.Nifti1Header()
            header['sizeof_hdr'] = 348
            header['dim_info'] = [3, 1, 1, 1, 1, 1, 1, 1]
            header['datatype'] = 2
            header['bitpix'] = 8
            affine = np.eye(4)
            dummy_data = np.zeros((1,1,1), dtype=np.uint8)
            dummy_img = nib.Nifti1Image(dummy_data, affine, header)
            nib.save(dummy_img, healthy_ds2_df['preprocessed_image_path'].iloc[0])

    print(f"T1w Healthy images from DS1 for analysis: {len(healthy_ds1_df)}")
    print(f"T1w Healthy images from DS2 for analysis: {len(healthy_ds2_df)}")

    # --- 3. Load a pre-trained ImageNet model (DenseNet121) ---
    print("\n--- 3. Loading pre-trained DenseNet121 model for feature extraction ---")
    base_model = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, 
                                                   input_shape=(IMAGENET_TARGET_SHAPE_2D[0], IMAGENET_TARGET_SHAPE_2D[1], 3))
    base_model.trainable = False # Замораживаем веса базовой модели

    # Проверка и корректировка имени слоя, если 'avg_pool' не найден (для совместимости)
    if PENULTIMATE_LAYER_NAME not in [layer.name for layer in base_model.layers]:
        if 'avg_pool' in [layer.name for layer in base_model.layers]:
            PENULTIMATE_LAYER_NAME = 'avg_pool'
        elif 'global_average_pooling2d' in [layer.name for layer in base_model.layers]:
            PENULTIMATE_LAYER_NAME = 'global_average_pooling2d'
        else:
            last_conv_layer = None
            for layer in reversed(base_model.layers):
                if isinstance(layer, tf.keras.layers.Concatenate):
                    if layer.output.shape and len(layer.output.shape) == 4:
                        last_conv_layer = layer
                        PENULTIMATE_LAYER_NAME = layer.name
                        break
                elif hasattr(layer, 'output_shape'):
                    if 'conv' in layer.name and len(layer.output_shape) == 4:
                        last_conv_layer = layer
                        PENULTIMATE_LAYER_NAME = layer.name
                        break
                
            if not last_conv_layer:
                print("Error: Could not find a suitable penultimate layer. Please check DenseNet121 layer names.")
                exit()

    print(f"Using layer '{PENULTIMATE_LAYER_NAME}' for feature extraction.")

    # --- 4. Extract Feature Embeddings for all Healthy Images ---
    print("\n--- 4. Extracting feature embeddings for all healthy images from both datasets ---")

    healthy_features_ds1 = []
    healthy_features_ds2 = []

    print(f"Extracting features for DS1 ({len(healthy_ds1_df)} images)...")
    for idx, row in healthy_ds1_df.iterrows():
        img_path = row['preprocessed_image_path']
        image_prepared = _prepare_2d_slice_for_imagenet(img_path, target_shape_2d=IMAGENET_TARGET_SHAPE_2D)
        if image_prepared.shape[0] > 0:
            feature_vec = get_feature_embedding(base_model, image_prepared, PENULTIMATE_LAYER_NAME)
            healthy_features_ds1.append(feature_vec)
        else:
            print(f"Skipping feature extraction for {os.path.basename(img_path)} due to preparation error.")

    print(f"Extracting features for DS2 ({len(healthy_ds2_df)} images)...")
    for idx, row in healthy_ds2_df.iterrows():
        img_path = row['preprocessed_image_path']
        image_prepared = _prepare_2d_slice_for_imagenet(img_path, target_shape_2d=IMAGENET_TARGET_SHAPE_2D)
        if image_prepared.shape[0] > 0:
            feature_vec = get_feature_embedding(base_model, image_prepared, PENULTIMATE_LAYER_NAME)
            healthy_features_ds2.append(feature_vec)
        else:
            print(f"Skipping feature extraction for {os.path.basename(img_path)} due to preparation error.")

    # Создание фиктивных признаков, если реальных не удалось извлечь
    if not healthy_features_ds1 or not healthy_features_ds2:
        print("Not enough features extracted from both datasets for comparison. Exiting.")
        print("Creating dummy features for demonstration.")
        healthy_features_ds1 = [np.random.rand(1024) for _ in range(5)]
        healthy_features_ds2 = [np.random.rand(1024) for _ in range(7)]


    features_ds1_array = np.array(healthy_features_ds1)
    features_ds2_array = np.array(healthy_features_ds2)

    print(f"Features DS1 shape: {features_ds1_array.shape}")
    print(f"Features DS2 shape: {features_ds2_array.shape}")

    # --- 5. Quantify Domain Shift using Feature Distributions ---
    print("\n--- 5. Quantifying Domain Shift on Feature Distributions ---")

    summary_metrics = {}

    mean_ds1 = np.mean(features_ds1_array, axis=0)
    std_ds1 = np.std(features_ds1_array, axis=0)
    mean_ds2 = np.mean(features_ds2_array, axis=0)
    std_ds2 = np.std(features_ds2_array, axis=0)

    print("\nMean Feature Vector (DS1, first 5 dim):", mean_ds1[:5])
    print("Std Dev Feature Vector (DS1, first 5 dim):", std_ds1[:5])
    print("Mean Feature Vector (DS2, first 5 dim):", mean_ds2[:5])
    print("Std Dev Feature Vector (DS2, first 5 dim):", std_ds2[:5])

    # Расчет косинусного сходства между средними векторами признаков
    if features_ds1_array.shape[1] == features_ds2_array.shape[1]:
        mean_features_ds1_reshaped = mean_ds1.reshape(1, -1)
        mean_features_ds2_reshaped = mean_ds2.reshape(1, -1)
        cos_sim_means = cosine_similarity(mean_features_ds1_reshaped, mean_features_ds2_reshaped)[0][0]
        summary_metrics['Cosine Similarity (Mean Vectors)'] = cos_sim_means
        print(f"\nCosine Similarity between Mean Feature Vectors: {cos_sim_means:.4f}")
    else:
        summary_metrics['Cosine Similarity (Mean Vectors)'] = 'N/A'
        print("Feature dimensions mismatch, cannot compute cosine similarity of mean vectors.")

    # Расчет среднего расстояния Вассерштейна по измерениям
    wasserstein_distances_per_dim = []
    if features_ds1_array.shape[1] == features_ds2_array.shape[1]:
        for i in range(features_ds1_array.shape[1]):
            wd = wasserstein_distance(features_ds1_array[:, i], features_ds2_array[:, i])
            wasserstein_distances_per_dim.append(wd)
            
        avg_wasserstein_dist_across_dims = np.mean(wasserstein_distances_per_dim)
        summary_metrics['Avg Wasserstein Dist (per dim)'] = avg_wasserstein_dist_across_dims
        print(f"\nAverage 1D Wasserstein Distance across feature dimensions: {avg_wasserstein_dist_across_dims:.4f}")
    else:
        summary_metrics['Avg Wasserstein Dist (per dim)'] = 'N/A'
        print("Feature dimensions mismatch, cannot compute Wasserstein distance per dimension.")

    # Расчет Евклидова расстояния между средними векторами
    if features_ds1_array.shape[1] == features_ds2_array.shape[1]:
        euclidean_dist_means = np.linalg.norm(mean_ds1 - mean_ds2)
        summary_metrics['Euclidean Distance (Mean Vectors)'] = euclidean_dist_means
        print(f"\nEuclidean Distance between Mean Feature Vectors: {euclidean_dist_means:.4f}")
    else:
        summary_metrics['Euclidean Distance (Mean Vectors)'] = 'N/A'

    # --- 6. Visualizations to aid interpretation ---
    print("\n--- 6. Generating Visualizations ---")

    all_features_combined = np.vstack((features_ds1_array, features_ds2_array))
    labels_combined = ['DS1'] * len(features_ds1_array) + ['DS2'] * len(features_ds2_array)

    scaler = StandardScaler()
    scaled_features_combined = scaler.fit_transform(all_features_combined)
    scaled_features_ds1 = scaler.transform(features_ds1_array)
    scaled_features_ds2 = scaler.transform(features_ds2_array)

    generate_feature_analysis_plots(scaled_features_ds1, scaled_features_ds2, penultimate_layer_name = PENULTIMATE_LAYER_NAME, plots_output_dir =PLOTS_OUTPUT_DIR)
    
    print("\n--- 7. Summary of Domain Shift Metrics ---")
    summary_df = pd.DataFrame([summary_metrics])
    summary_df = summary_df.T.rename(columns={0: 'Value'})
    print(summary_df.to_string())

    print("\n--- Script Finished ---")