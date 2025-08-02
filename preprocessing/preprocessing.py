import SimpleITK as sitk  # Для нормализации интенсивности и коррекции поля смещения
import ants  # Для регистрации изображений
from nipype.interfaces.fsl import BET  # Для удаления черепа с помощью FSL
import os  # Для работы с файловой системой
import glob  # Для поиска файлов по шаблону

# === Функция коррекции поля смещения ===
def bias_field_correction(input_img, out_path):

    # Загрузка входного изображения в формате SimpleITK и приведение к типу float32
    raw_img_sitk = sitk.ReadImage(input_img, sitk.sitkFloat32)

    # Пороговая фильтрация с методом Li для выделения маски головы (0 - фон, 1 - мозг) за счёт минимазации энтропии
    head_mask = sitk.LiThreshold(raw_img_sitk, 0, 1)

    # Уменьшение размера изображения для ускорения обработки (ShrinkFactor)
    shrinkFactor = 4 # берём каждый 4-й пиксель по всем осям
    inputImage = sitk.Shrink(raw_img_sitk, [shrinkFactor] * raw_img_sitk.GetDimension())
    maskImage = sitk.Shrink(head_mask, [shrinkFactor] * raw_img_sitk.GetDimension())

    # Создание фильтра коррекции поля смещения (N4ITK)
    bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected = bias_corrector.Execute(inputImage, maskImage)

    # Получение логарифмического поля смещения
    log_bias_field = bias_corrector.GetLogBiasFieldAsImage(raw_img_sitk)
    # Коррекция изображения по логарифмическому полю смещения
    corrected_image_full_resolution = raw_img_sitk / sitk.Exp(log_bias_field)

    sitk.WriteImage(corrected_image_full_resolution, out_path)  # Сохраняем результат

# === Функция нормализации интенсивности ===
def intensity_normalization(input_img, out_path):

    # Загрузка шаблона MNI152 (стандартное изображение мозга)
    template_img_path = "/media/storage/roppert/mni_icbm152_t1_tal_nlin_sym_09a.nii"
    template_img_sitk = sitk.ReadImage(template_img_path, sitk.sitkFloat32)
    template_img_sitk = sitk.DICOMOrient(template_img_sitk, 'RPS')  # Переориентация изображения по стандарту

    # Загрузка входного изображения
    raw_img_sitk = sitk.ReadImage(input_img, sitk.sitkFloat32)

    # Применение гистограммного выравнивания для нормализации интенсивности
    transformed = sitk.HistogramMatching(raw_img_sitk, template_img_sitk)

    sitk.WriteImage(transformed, out_path)  # Сохраняем результат

# === Функция регистрации изображения ===
def registration(input_img, out_path):

    # Загрузка шаблона MNI152
    template_img_path = "/media/storage/roppert/mni_icbm152_t1_tal_nlin_sym_09a.nii"
    template_img_ants = ants.image_read(template_img_path, reorient='IAL')

    # Загрузка входного изображения
    raw_img_ants = ants.image_read(input_img, reorient='IAL')

    # Регистрация изображения по шаблону с использованием трансформации 'SyN' (одна из лучших для мозга)
    transformation = ants.registration(
        fixed=template_img_ants,
        moving=raw_img_ants,
        type_of_transform='SyN',
        verbose=False
    )
     # Извлечение зарегистрированного изображения из результата
    registered_img_ants = transformation['warpedmovout']
    registered_img_ants.to_file(out_path)  # Сохраняем результат

# === Функция удаления черепа ===
def skull_stripping(input_img, out_path):

    # Создание экземпляра BET-инструмента
    bet = BET()
    bet.inputs.in_file = input_img  # Входной файл
    bet.inputs.out_file = out_path  # Выходной файл
    bet.inputs.frac = 0.5  # Параметр степени усечения
    bet.inputs.mask = True  # Создание маски мозга
    bet.inputs.robust = True  # Улучшение стабильности алгоритма

    bet.run()  # Запуск инструмента BET

# === Основная функция обработки всех файлов ===
def process_nifti_files(input_root, output_root):
    os.makedirs(output_root, exist_ok=True)  # Создаём выходную директорию, если её нет

    # Цикл по всем пациентам в корневой папке
    for patient_folder in sorted(os.listdir(input_root)):
        patient_input_path = os.path.join(input_root, patient_folder)
        if not os.path.isdir(patient_input_path):
            continue

        patient_output_path = os.path.join(output_root, patient_folder)
        os.makedirs(patient_output_path, exist_ok=True)

        nii_files = sorted(glob.glob(os.path.join(patient_input_path, "*.nii.gz")))

        print(f"[INFO] Обработка пациента: {patient_folder}")

        # 1. Нормализация интенсивности
        norm_files = []
        intensity_norm_path = os.path.join(patient_output_path, "intensity_normalized")
        os.makedirs(intensity_norm_path, exist_ok=True)
        print("[INFO] Выполняется Intensity Normalization...")
        for file in nii_files:
            filename = os.path.basename(file)
            out_file = os.path.join(intensity_norm_path, filename)
            intensity_normalization(file, out_file)
            norm_files.append(out_file)

        # 2. Коррекция поля смещения
        corrected_files = []
        bias_corrected_path = os.path.join(patient_output_path, "bias_field_corrected")
        os.makedirs(bias_corrected_path, exist_ok=True)
        print("[INFO] Выполняется Bias Field Correction...")
        for file in norm_files:
            filename = os.path.basename(file)
            out_file = os.path.join(bias_corrected_path, filename)
            bias_field_correction(file, out_file)
            corrected_files.append(out_file)

        # 3. Регистрация
        reg_files = []
        registered_path = os.path.join(patient_output_path, "registered")
        os.makedirs(registered_path, exist_ok=True)
        print("[INFO] Выполняется Registration...")
        for file in corrected_files:
            filename = os.path.basename(file)
            out_file = os.path.join(registered_path, filename)
            registration(file, out_file)
            reg_files.append(out_file)

        # 4. Skull Stripping (удаление черепа)
        skull_stripped_path = os.path.join(patient_output_path, "skull_stripped")
        os.makedirs(skull_stripped_path, exist_ok=True)
        print("[INFO] Выполняется Skull Stripping...")
        for file in reg_files:
            filename = os.path.basename(file)
            out_file = os.path.join(skull_stripped_path, filename)
            skull_stripping(file, out_file)

        print(f"[INFO] Обработка пациента {patient_folder} завершена.\n")

if __name__ == "__main__":
    input_directory = "nifti_data"  # Путь к папке с входными изображениями
    output_directory = "test_processed_data"  # Путь для сохранения результатов
    process_nifti_files(input_directory, output_directory)