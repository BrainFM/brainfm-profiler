import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import ttest_ind

def generate_feature_analysis_plots(features_ds1_array, features_ds2_array, penultimate_layer_name, plots_output_dir):
    """
    Генерирует различные визуализации и выполняет статистический анализ
    распределений признаков из предпоследнего слоя для двух наборов данных.

    Args:
        features_ds1_array (np.ndarray): Признаки из предпоследнего слоя для Датасета 1.
                                         Ожидается двумерный массив (количество_сэмплов, количество_признаков).
        features_ds2_array (np.ndarray): Признаки из предпоследнего слоя для Датасета 2.
                                         Ожидается двумерный массив (количество_сэмплов, количество_признаков).
        penultimate_layer_name (str): Имя предпоследнего слоя модели (для заголовков графиков).
        plots_output_dir (str): Путь к директории для сохранения графиков.
    """
    if not os.path.exists(plots_output_dir):
        os.makedirs(plots_output_dir)
        print(f"Создана директория для сохранения графиков: {plots_output_dir}")

    print("\n--- 6. Генерация Визуализаций ---")

    # Проверка на соответствие размерностей признаков
    if features_ds1_array.shape[1] != features_ds2_array.shape[1]:
        print("Ошибка: Размерности признаков (количество признаков) для DS1 и DS2 не совпадают. Визуализации распределений и статистический анализ будут ограничены.")
        # Для продолжения визуализаций, которые не требуют одинаковых размерностей,
        # можно обрезать или использовать только те признаки, которые есть в обоих.
        # Для простоты, многие из следующих шагов предполагают одинаковые размерности.
        # Если это критично, нужно решить, как обрабатывать. Пока ограничимся выводом ошибки.
        return # Выходим из функции, если размерности несовместимы для большинства анализов

    all_features_combined = np.vstack((features_ds1_array, features_ds2_array))
    labels_combined = ['DS1'] * len(features_ds1_array) + ['DS2'] * len(features_ds2_array)

    # Масштабирование признаков
    scaler = StandardScaler()
    scaled_features_combined = scaler.fit_transform(all_features_combined)
    # Масштабируем отдельные датасеты, используя тот же скейлер, обученный на объединенных данных
    scaled_features_ds1 = scaler.transform(features_ds1_array)
    scaled_features_ds2 = scaler.transform(features_ds2_array)

    # --- Гистограммы распределений признаков ---
    num_hist_features = min(5, features_ds1_array.shape[1]) # Отображаем гистограммы для первых 5 признаков
    if num_hist_features > 0:
        plt.figure(figsize=(15, 3 * num_hist_features))
        plt.suptitle(f'Гистограммы распределений признаков ({penultimate_layer_name})', fontsize=16)
        for i in range(num_hist_features):
            plt.subplot(num_hist_features, 2, 2*i + 1)
            sns.histplot(scaled_features_ds1[:, i], kde=True, color='skyblue', label='DS1', stat='density')
            plt.title(f'Признак {i+1} (DS1)')
            plt.xlabel('Масштабированное значение признака')
            plt.ylabel('Плотность')
            plt.legend()
            
            plt.subplot(num_hist_features, 2, 2*i + 2)
            sns.histplot(scaled_features_ds2[:, i], kde=True, color='salmon', label='DS2', stat='density')
            plt.title(f'Признак {i+1} (DS2)')
            plt.xlabel('Масштабированное значение признака')
            plt.ylabel('Плотность')
            plt.legend()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(plots_output_dir, 'histograms_feature_distributions.png'))
        plt.close()
    else:
        print("Недостаточно признаков для построения гистограмм.")

    # --- Box Plot распределений признаков ---
    if num_hist_features > 0:
        plt.figure(figsize=(15, 3 * num_hist_features))
        plt.suptitle(f'Box Plots распределений признаков ({penultimate_layer_name})', fontsize=16)
        for i in range(num_hist_features):
            feature_data = pd.DataFrame({
                'Value': np.concatenate([scaled_features_ds1[:, i], scaled_features_ds2[:, i]]),
                'Dataset': ['DS1'] * len(scaled_features_ds1) + ['DS2'] * len(scaled_features_ds2)
            })
            plt.subplot(num_hist_features, 1, i + 1)
            sns.boxplot(x='Dataset', y='Value', data=feature_data, palette={'DS1': 'skyblue', 'DS2': 'salmon'})
            plt.title(f'Признак {i+1}')
            plt.ylabel('Масштабированное значение признака')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(plots_output_dir, 'boxplots_feature_distributions.png'))
        plt.close()
    else:
        print("Недостаточно признаков для построения Box Plots.")


    # --- Тепловые карты корреляции признаков ---
    # Строим только если количество признаков не слишком велико
    if all_features_combined.shape[1] < 200: 
        plt.figure(figsize=(12, 10))
        sns.heatmap(pd.DataFrame(scaled_features_ds1).corr(), cmap='viridis', annot=False, fmt=".2f", cbar=True)
        plt.title(f'Тепловая карта корреляции признаков (DS1) для слоя {penultimate_layer_name}')
        plt.savefig(os.path.join(plots_output_dir, 'correlation_heatmap_ds1.png'))
        plt.close()

        plt.figure(figsize=(12, 10))
        sns.heatmap(pd.DataFrame(scaled_features_ds2).corr(), cmap='viridis', annot=False, fmt=".2f", cbar=True)
        plt.title(f'Тепловая карта корреляции признаков (DS2) для слоя {penultimate_layer_name}')
        plt.savefig(os.path.join(plots_output_dir, 'correlation_heatmap_ds2.png'))
        plt.close()
    else:
        print(f"Пропуск построения тепловых карт корреляции из-за высокой размерности признаков ({all_features_combined.shape[1]}).")


    # --- PCA (Анализ Главных Компонент) ---
    # Проверка на достаточное количество данных для PCA (минимум 2 образца и 2 признака)
    if all_features_combined.shape[0] > 1 and all_features_combined.shape[1] > 1: 
        try:
            pca = PCA(n_components=2)
            principal_components = pca.fit_transform(scaled_features_combined)
            pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
            pca_df['Dataset'] = labels_combined

            plt.figure(figsize=(10, 8))
            sns.scatterplot(x='PC1', y='PC2', hue='Dataset', data=pca_df, alpha=0.7, s=50, palette={'DS1': 'blue', 'DS2': 'red'})
            
            # Отображение центроидов
            centroid_ds1 = pca.transform(scaler.transform(np.mean(features_ds1_array, axis=0).reshape(1, -1)))
            centroid_ds2 = pca.transform(scaler.transform(np.mean(features_ds2_array, axis=0).reshape(1, -1)))
            plt.scatter(centroid_ds1[0, 0], centroid_ds1[0, 1], marker='X', s=300, color='blue', edgecolor='black', linewidth=2, label='Центроид DS1', zorder=5)
            plt.scatter(centroid_ds2[0, 0], centroid_ds2[0, 1], marker='X', s=300, color='red', edgecolor='black', linewidth=2, label='Центроид DS2', zorder=5)

            plt.title(f'PCA вложений признаков глиомы ({penultimate_layer_name})')
            plt.xlabel(f'Главная Компонента 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
            plt.ylabel(f'Главная Компонента 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(plots_output_dir, 'pca_feature_embeddings.png'))
            plt.close()
        except ValueError as e:
            print(f"Ошибка при выполнении PCA: {e}. Проверьте данные.")
    else:
        print("Недостаточно точек данных или признаков для выполнения PCA для визуализации.")


    # --- t-SNE (t-distributed Stochastic Neighbor Embedding) ---
    # Проверка на достаточное количество данных для t-SNE (рекомендуется > 30 образцов)
    # Perplexity должна быть меньше, чем количество образцов - 1
    if all_features_combined.shape[0] > 30 and all_features_combined.shape[1] > 1:
        print("Запуск t-SNE (это может занять некоторое время)...")
        try:
            tsne = TSNE(n_components=2, random_state=42,
                        perplexity=min(30, len(all_features_combined) - 1)) # Removed n_iter
            tsne_components = tsne.fit_transform(scaled_features_combined)
            tsne_df = pd.DataFrame(data=tsne_components, columns=['TSNE1', 'TSNE2'])
            tsne_df['Dataset'] = labels_combined

            plt.figure(figsize=(10, 8))
            sns.scatterplot(x='TSNE1', y='TSNE2', hue='Dataset', data=tsne_df, alpha=0.7, s=50, palette={'DS1': 'blue', 'DS2': 'red'})
            plt.title(f't-SNE вложений признаков глиомы ({penultimate_layer_name})')
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(plots_output_dir, 'tsne_feature_embeddings.png'))
            plt.close()
        except ValueError as e:
            print(f"Ошибка при выполнении t-SNE: {e}. Проверьте данные или параметры.")
    else:
        print("Недостаточно точек данных для визуализации t-SNE (требуется как минимум 30 сэмплов).")

    # --- Статистический анализ различий по признакам (T-критерий Стьюдента) ---
    if features_ds1_array.shape[1] == features_ds2_array.shape[1] and features_ds1_array.shape[0] > 1 and features_ds2_array.shape[0] > 1:
        p_values = []
        for i in range(features_ds1_array.shape[1]):
            # Используем Welch's t-test (equal_var=False) для несмещенных выборок
            # Если данные нормально распределены, это хороший выбор.
            # Если нет, то можно рассмотреть непараметрические тесты, но t-тест часто робастен для достаточно больших выборок.
            stat, p = ttest_ind(features_ds1_array[:, i], features_ds2_array[:, i], equal_var=False)
            p_values.append(p)
            
        p_values_array = np.array(p_values)
        total_features = features_ds1_array.shape[1]
            
        print(f"\n--- Статистический анализ различий по всем {total_features} признакам (T-критерий Стьюдента Уэлча) ---")

        # Нескорректированное p-значение
        significant_features_05 = np.sum(p_values_array < 0.05)
        print(f"Всего признаков: {total_features}")
        print(f"Признаков с p-значением < 0.05 (некорректированное): {significant_features_05} ({significant_features_05 / total_features:.2%})")

        # Коррекция Бонферрони для множественных сравнений
        if total_features > 0:
            bonferroni_alpha = 0.05 / total_features
            significant_features_bonferroni = np.sum(p_values_array < bonferroni_alpha)
            print(f"Признаков с p-значением < {bonferroni_alpha:.2e} (с поправкой Бонферрони): {significant_features_bonferroni} ({significant_features_bonferroni / total_features:.2%})")
            print("Высокий процент признаков, статистически значимых даже после поправки Бонферрони, может указывать на систематические различия между наборами данных.")
        else:
            print("Нет признаков для статистического анализа.")

        # --- Гистограммы наиболее различающихся признаков ---
        num_top_features_to_display = min(20, total_features)     
        # Сортируем p-значения по возрастанию, чтобы найти наиболее различающиеся признаки
        top_dissimilar_indices = np.argsort(p_values_array)[:num_top_features_to_display]

        if num_top_features_to_display > 0:
            plt.figure(figsize=(15, 4 * num_top_features_to_display))     
            plt.suptitle(f'Распределения Топ {num_top_features_to_display} наиболее различающихся признаков ({penultimate_layer_name})', fontsize=16)
            for i, idx in enumerate(top_dissimilar_indices):
                feature_data = pd.DataFrame({
                    'Value': np.concatenate([scaled_features_ds1[:, idx], scaled_features_ds2[:, idx]]),
                    'Dataset': ['DS1'] * len(scaled_features_ds1) + ['DS2'] * len(scaled_features_ds2)
                })
                plt.subplot(num_top_features_to_display, 1, i + 1)     
                sns.histplot(data=feature_data, x='Value', hue='Dataset', kde=True, stat='density', common_norm=False, palette={'DS1': 'skyblue', 'DS2': 'salmon'})
                plt.title(f'Признак {idx+1} (p-значение: {p_values_array[idx]:.2e})')
                plt.xlabel('Масштабированное значение признака')
                plt.ylabel('Плотность')
                plt.legend(title='Набор данных')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(plots_output_dir, 'top_dissimilar_features_histograms.png'))
            plt.close()
        else:
            print("Недостаточно признаков для отображения наиболее различающихся гистограмм.")
    else:
        print("Невозможно выполнить статистический анализ различий по признакам из-за несовпадения размерностей или недостаточного количества выборок.")