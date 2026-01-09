"""
Фрактальная граница обучаемости нейросетей
Реализация экспериментов для курсовой работы

Автор: Бухаров Дмитрий Иванович
Группа: 25.Б21
Научный руководитель: д.ф.-м.н., проф. Мокаев Т.Н.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.morphology import skeletonize
from tqdm import tqdm
import json
import os

# Установка точности float64 для наблюдения мелкомасштабной структуры
torch.set_default_dtype(torch.float64)

# Для воспроизводимости результатов
np.random.seed(42)
torch.manual_seed(42)

class SimpleNetwork:
    """
    Однослойная нейронная сеть с настраиваемой функцией активации
    """
    def __init__(self, input_dim=16, hidden_dim=16, output_dim=16, activation='tanh'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation
        
        # Инициализация весов (без bias)
        self.W0 = torch.randn(hidden_dim, input_dim, dtype=torch.float64) * 0.1
        self.W1 = torch.randn(output_dim, hidden_dim, dtype=torch.float64) * 0.1
        
    def forward(self, X):
        """Прямой проход"""
        h = X @ self.W0.T
        
        # Применение функции активации
        if self.activation == 'tanh':
            h = torch.tanh(h)
        elif self.activation == 'relu':
            h = torch.relu(h)
        elif self.activation == 'linear':
            pass  # Без активации
        
        out = h @ self.W1.T
        return out
    
    def compute_loss(self, X, Y):
        """Вычисление MSE loss"""
        out = self.forward(X)
        return ((out - Y) ** 2).mean()
    
    def gradient_step(self, X, Y, lr0, lr1):
        """Один шаг градиентного спуска с раздельными learning rates"""
        # Forward pass
        h = X @ self.W0.T
        if self.activation == 'tanh':
            h_act = torch.tanh(h)
        elif self.activation == 'relu':
            h_act = torch.relu(h)
        else:
            h_act = h
            
        out = h_act @ self.W1.T
        loss = ((out - Y) ** 2).mean()
        
        # Backward pass (ручное вычисление градиентов)
        grad_out = 2 * (out - Y) / (Y.shape[0] * Y.shape[1])
        
        # Градиент для W1
        grad_W1 = grad_out.T @ h_act
        
        # Градиент для W0
        grad_h_act = grad_out @ self.W1
        if self.activation == 'tanh':
            grad_h = grad_h_act * (1 - h_act ** 2)
        elif self.activation == 'relu':
            grad_h = grad_h_act * (h > 0).float()
        else:
            grad_h = grad_h_act
            
        grad_W0 = grad_h.T @ X
        
        # Обновление весов
        self.W0 = self.W0 - lr0 * grad_W0
        self.W1 = self.W1 - lr1 * grad_W1
        
        return loss.item()


def train_single_config(lr0, lr1, X, Y, activation='tanh', max_steps=1000, 
                       convergence_threshold=1.0, divergence_threshold=1e6):
    """
    Обучение одной конфигурации с заданными learning rates
    
    Returns:
        status: 'converged', 'diverged', или 'timeout'
        final_step: номер последнего шага
        loss_sum: сумма нормализованных loss (для визуализации)
    """
    net = SimpleNetwork(activation=activation)
    
    # Первый loss для нормализации
    initial_loss = net.compute_loss(X, Y)
    
    loss_history = []
    
    for step in range(max_steps):
        loss = net.gradient_step(X, Y, lr0, lr1)
        
        # Нормализация к начальному loss
        normalized_loss = loss / initial_loss
        loss_history.append(normalized_loss)
        
        # Проверка на расхождение
        if np.isnan(loss) or np.isinf(loss) or loss > divergence_threshold:
            return 'diverged', step, sum(1.0 / max(l, 1e-10) for l in loss_history)
        
        # Ранняя остановка при стабилизации
        if step > 50 and np.std(loss_history[-20:]) < 1e-8:
            break
    
    # Проверка на сходимость (среднее последних 20 итераций)
    if len(loss_history) >= 20:
        if np.mean(loss_history[-20:]) < convergence_threshold:
            return 'converged', step, sum(loss_history)
    
    return 'timeout', max_steps, sum(loss_history)


def scan_hyperparameter_space(lr0_range, lr1_range, X, Y, activation='tanh',
                              max_steps=1000, resolution=512):
    """
    Сканирование пространства гиперпараметров (lr0, lr1)
    
    Returns:
        grid: массив с результатами (положительные = сходимость, отрицательные = расхождение)
        lr0_grid, lr1_grid: сетки значений learning rates
    """
    print(f"Сканирование пространства {resolution}x{resolution} с активацией {activation}...")
    
    # Создание логарифмической сетки
    lr0_values = np.logspace(np.log10(lr0_range[0]), np.log10(lr0_range[1]), resolution)
    lr1_values = np.logspace(np.log10(lr1_range[0]), np.log10(lr1_range[1]), resolution)
    
    lr0_grid, lr1_grid = np.meshgrid(lr0_values, lr1_values)
    
    results = np.zeros((resolution, resolution))
    
    total = resolution * resolution
    with tqdm(total=total) as pbar:
        for i in range(resolution):
            for j in range(resolution):
                status, step, measure = train_single_config(
                    lr0_grid[i, j], lr1_grid[i, j], X, Y,
                    activation=activation, max_steps=max_steps
                )
                
                # Кодирование результата:
                # положительные значения = сходимость (меньше = быстрее)
                # отрицательные значения = расхождение (ближе к 0 = быстрее)
                if status == 'converged':
                    results[i, j] = measure
                elif status == 'diverged':
                    results[i, j] = -measure
                else:  # timeout
                    results[i, j] = measure * 2  # помечаем как медленную сходимость
                
                pbar.update(1)
    
    return results, lr0_grid, lr1_grid


def extract_boundary(results):
    """
    Извлечение границы между областями сходимости и расхождения
    """
    # Бинаризация: True = сходимость, False = расхождение
    binary = results > 0
    
    # Морфологическое выделение границы
    dilated = binary_dilation(binary)
    eroded = binary_erosion(binary)
    boundary = dilated ^ eroded
    
    # Скелетонизация для получения тонкой границы
    boundary_thin = skeletonize(boundary)
    
    return boundary_thin.astype(np.uint8)


def compute_box_counting_dimension(boundary):
    """
    Вычисление фрактальной размерности методом box-counting
    """
    # Диапазон размеров ячеек
    max_box_size = min(boundary.shape) // 4
    min_box_size = 2
    
    box_sizes = []
    box_counts = []
    
    # Перебираем степени двойки
    box_size = min_box_size
    while box_size <= max_box_size:
        # Подсчет непустых ячеек
        count = 0
        for i in range(0, boundary.shape[0], box_size):
            for j in range(0, boundary.shape[1], box_size):
                box = boundary[i:i+box_size, j:j+box_size]
                if box.sum() > 0:
                    count += 1
        
        box_sizes.append(box_size)
        box_counts.append(count)
        box_size *= 2
    
    # Линейная регрессия в log-log пространстве
    if len(box_sizes) > 2:
        log_sizes = np.log(box_sizes)
        log_counts = np.log(box_counts)
        
        # y = kx + b => log(N) = -D * log(ε) + b
        coeffs = np.polyfit(log_sizes, log_counts, 1)
        fractal_dim = -coeffs[0]
        
        return fractal_dim, box_sizes, box_counts
    else:
        return None, box_sizes, box_counts


def visualize_landscape(results, lr0_grid, lr1_grid, save_path=None, title=""):
    """
    Визуализация ландшафта обучаемости
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Разделение на сходящиеся и расходящиеся
    converged = np.where(results > 0, results, np.nan)
    diverged = np.where(results < 0, -results, np.nan)
    
    # Нормализация для цветовой карты
    if not np.all(np.isnan(converged)):
        conv_norm = converged / np.nanmax(converged)
        im1 = ax.contourf(lr0_grid, lr1_grid, conv_norm, levels=20, 
                         cmap='Blues_r', alpha=0.8)
    
    if not np.all(np.isnan(diverged)):
        div_norm = diverged / np.nanmax(diverged)
        im2 = ax.contourf(lr0_grid, lr1_grid, div_norm, levels=20,
                         cmap='Reds_r', alpha=0.8)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Learning Rate η₀ (input layer)', fontsize=12)
    ax.set_ylabel('Learning Rate η₁ (output layer)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Добавляем colorbar
    plt.colorbar(im1 if not np.all(np.isnan(converged)) else im2, ax=ax,
                label='Normalized convergence/divergence measure')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Сохранено: {save_path}")
    
    plt.close()


def plot_box_counting(box_sizes, box_counts, fractal_dim, save_path=None):
    """
    График box-counting для визуализации фрактальной размерности
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    log_sizes = np.log(box_sizes)
    log_counts = np.log(box_counts)
    
    ax.scatter(log_sizes, log_counts, s=100, alpha=0.7, edgecolors='black')
    
    # Линия регрессии
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    fit_line = np.poly1d(coeffs)
    ax.plot(log_sizes, fit_line(log_sizes), 'r--', linewidth=2,
           label=f'Slope = {-coeffs[0]:.3f}')
    
    ax.set_xlabel('log(box size ε)', fontsize=12)
    ax.set_ylabel('log(number of boxes N(ε))', fontsize=12)
    ax.set_title(f'Box-Counting Method\nFractal Dimension D = {fractal_dim:.3f}', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Сохранено: {save_path}")
    
    plt.close()


def generate_zoom_sequence(center_lr0, center_lr1, X, Y, activation='tanh',
                          num_zooms=7, zoom_factor=2, base_resolution=256):
    """
    Генерация последовательности зумов для демонстрации фрактальной структуры
    """
    results = []
    dimensions = []
    
    current_width_lr0 = 1.0
    current_width_lr1 = 1.0
    
    for zoom_level in range(num_zooms):
        print(f"\nУровень зума {zoom_level + 1}/{num_zooms}")
        
        # Определение диапазона для текущего уровня
        lr0_range = [
            max(1e-6, center_lr0 - current_width_lr0 / 2),
            center_lr0 + current_width_lr0 / 2
        ]
        lr1_range = [
            max(1e-6, center_lr1 - current_width_lr1 / 2),
            center_lr1 + current_width_lr1 / 2
        ]
        
        # Сканирование
        result_grid, lr0_grid, lr1_grid = scan_hyperparameter_space(
            lr0_range, lr1_range, X, Y,
            activation=activation,
            resolution=base_resolution
        )
        
        # Извлечение границы и вычисление размерности
        boundary = extract_boundary(result_grid)
        dim, sizes, counts = compute_box_counting_dimension(boundary)
        
        if dim is not None:
            dimensions.append(dim)
            print(f"Фрактальная размерность: {dim:.3f}")
        
        results.append({
            'zoom_level': zoom_level,
            'lr0_range': lr0_range,
            'lr1_range': lr1_range,
            'results': result_grid,
            'lr0_grid': lr0_grid,
            'lr1_grid': lr1_grid,
            'boundary': boundary,
            'fractal_dim': dim,
            'box_sizes': sizes,
            'box_counts': counts
        })
        
        # Уменьшаем ширину для следующего зума
        current_width_lr0 /= zoom_factor
        current_width_lr1 /= zoom_factor
    
    # Медианная размерность по всем уровням
    if dimensions:
        median_dim = np.median(dimensions)
        print(f"\n{'='*50}")
        print(f"Медианная фрактальная размерность: {median_dim:.3f}")
        print(f"Диапазон: [{min(dimensions):.3f}, {max(dimensions):.3f}]")
        print(f"{'='*50}")
    
    return results, median_dim if dimensions else None


def toy_quadratic_regression(lr_range, num_points=1000):
    """
    Игрушечная задача квадратичной регрессии для демонстрации переходов
    в динамике градиентного спуска
    """
    # Простая квадратичная задача: min ||Ax - b||^2
    A = torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    b = torch.tensor([1.0, 1.0], dtype=torch.float64)
    
    results = {
        'learning_rates': [],
        'final_loss': [],
        'trajectory_type': [],
        'loss_history': []
    }
    
    for lr in np.linspace(lr_range[0], lr_range[1], num_points):
        x = torch.zeros(2, dtype=torch.float64)
        losses = []
        
        for step in range(200):
            residual = A @ x - b
            loss = (residual ** 2).sum()
            losses.append(loss.item())
            
            grad = 2 * A.T @ residual
            x = x - lr * grad
            
            if torch.isnan(x).any() or loss > 1e10:
                break
        
        # Классификация типа траектории
        if len(losses) < 200:
            traj_type = 'diverged'
        elif np.std(losses[-20:]) < 1e-10:
            if all(losses[i] >= losses[i+1] for i in range(len(losses)-1)):
                traj_type = 'monotonic'
            else:
                traj_type = 'catapult'
        elif len(set(np.round(losses[-20:], 6))) <= 3:
            traj_type = 'periodic'
        else:
            traj_type = 'chaotic'
        
        results['learning_rates'].append(lr)
        results['final_loss'].append(losses[-1] if losses else np.inf)
        results['trajectory_type'].append(traj_type)
        results['loss_history'].append(losses)
    
    return results


def main():
    """
    Основная функция для запуска всех экспериментов
    """
    print("="*70)
    print("Фрактальная граница обучаемости нейросетей")
    print("Курсовая работа")
    print("Автор: Бухаров Дмитрий Иванович, группа 25.Б21")
    print("="*70)
    
    # Создание выходной директории
    os.makedirs('/mnt/user-data/outputs/figures', exist_ok=True)
    os.makedirs('/mnt/user-data/outputs/data', exist_ok=True)
    
    # Генерация данных
    print("\nГенерация тренировочных данных...")
    n_samples = 32
    input_dim = 16
    output_dim = 16
    
    X = torch.randn(n_samples, input_dim, dtype=torch.float64)
    Y = torch.randn(n_samples, output_dim, dtype=torch.float64)
    
    # Эксперимент 1: Базовый скан с tanh
    print("\n" + "="*70)
    print("ЭКСПЕРИМЕНТ 1: Базовое сканирование (tanh активация)")
    print("="*70)
    
    results_tanh, lr0_grid, lr1_grid = scan_hyperparameter_space(
        lr0_range=[0.001, 10.0],
        lr1_range=[0.001, 10.0],
        X=X, Y=Y,
        activation='tanh',
        resolution=512
    )
    
    visualize_landscape(results_tanh, lr0_grid, lr1_grid,
                       save_path='/mnt/user-data/outputs/figures/landscape_tanh.png',
                       title='Trainability Landscape (tanh activation)')
    
    boundary_tanh = extract_boundary(results_tanh)
    dim_tanh, sizes_tanh, counts_tanh = compute_box_counting_dimension(boundary_tanh)
    
    if dim_tanh:
        print(f"\nФрактальная размерность (tanh): {dim_tanh:.3f}")
        plot_box_counting(sizes_tanh, counts_tanh, dim_tanh,
                         save_path='/mnt/user-data/outputs/figures/boxcount_tanh.png')
    
    # Эксперимент 2: ReLU активация
    print("\n" + "="*70)
    print("ЭКСПЕРИМЕНТ 2: ReLU активация")
    print("="*70)
    
    results_relu, _, _ = scan_hyperparameter_space(
        lr0_range=[0.001, 10.0],
        lr1_range=[0.001, 10.0],
        X=X, Y=Y,
        activation='relu',
        resolution=512
    )
    
    visualize_landscape(results_relu, lr0_grid, lr1_grid,
                       save_path='/mnt/user-data/outputs/figures/landscape_relu.png',
                       title='Trainability Landscape (ReLU activation)')
    
    boundary_relu = extract_boundary(results_relu)
    dim_relu, sizes_relu, counts_relu = compute_box_counting_dimension(boundary_relu)
    
    if dim_relu:
        print(f"\nФрактальная размерность (ReLU): {dim_relu:.3f}")
        plot_box_counting(sizes_relu, counts_relu, dim_relu,
                         save_path='/mnt/user-data/outputs/figures/boxcount_relu.png')
    
    # Эксперимент 3: Линейная сеть
    print("\n" + "="*70)
    print("ЭКСПЕРИМЕНТ 3: Линейная сеть (без активации)")
    print("="*70)
    
    results_linear, _, _ = scan_hyperparameter_space(
        lr0_range=[0.001, 10.0],
        lr1_range=[0.001, 10.0],
        X=X, Y=Y,
        activation='linear',
        resolution=512
    )
    
    visualize_landscape(results_linear, lr0_grid, lr1_grid,
                       save_path='/mnt/user-data/outputs/figures/landscape_linear.png',
                       title='Trainability Landscape (Linear network)')
    
    boundary_linear = extract_boundary(results_linear)
    dim_linear, sizes_linear, counts_linear = compute_box_counting_dimension(boundary_linear)
    
    if dim_linear:
        print(f"\nФрактальная размерность (Linear): {dim_linear:.3f}")
        plot_box_counting(sizes_linear, counts_linear, dim_linear,
                         save_path='/mnt/user-data/outputs/figures/boxcount_linear.png')
    
    # Эксперимент 4: Zoom sequence
    print("\n" + "="*70)
    print("ЭКСПЕРИМЕНТ 4: Последовательность зумов")
    print("="*70)
    
    # Выбираем точку на границе из результатов tanh
    center_lr0, center_lr1 = 0.5, 0.5
    
    zoom_results, median_dim = generate_zoom_sequence(
        center_lr0=center_lr0,
        center_lr1=center_lr1,
        X=X, Y=Y,
        activation='tanh',
        num_zooms=7,
        zoom_factor=2,
        base_resolution=256
    )
    
    # Визуализация зумов
    for i, zr in enumerate(zoom_results):
        visualize_landscape(
            zr['results'], zr['lr0_grid'], zr['lr1_grid'],
            save_path=f'/mnt/user-data/outputs/figures/zoom_level_{i+1}.png',
            title=f'Zoom Level {i+1} (range: [{zr["lr0_range"][0]:.2e}, {zr["lr0_range"][1]:.2e}])'
        )
    
    # Эксперимент 5: Квадратичная регрессия
    print("\n" + "="*70)
    print("ЭКСПЕРИМЕНТ 5: Динамика градиентного спуска (квадратичная регрессия)")
    print("="*70)
    
    quad_results = toy_quadratic_regression(lr_range=[0.0, 3.0], num_points=1000)
    
    # Визуализация фазового перехода
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    lrs = quad_results['learning_rates']
    final_losses = quad_results['final_loss']
    types = quad_results['trajectory_type']
    
    # График 1: Финальный loss vs learning rate
    colors = {'monotonic': 'blue', 'catapult': 'green', 'periodic': 'orange', 
              'chaotic': 'red', 'diverged': 'black'}
    
    for ttype in colors.keys():
        mask = [t == ttype for t in types]
        if any(mask):
            ax1.scatter([lrs[i] for i, m in enumerate(mask) if m],
                       [final_losses[i] for i, m in enumerate(mask) if m],
                       c=colors[ttype], label=ttype, alpha=0.6, s=20)
    
    ax1.set_xlabel('Learning Rate η', fontsize=12)
    ax1.set_ylabel('Final Loss', fontsize=12)
    ax1.set_title('Phase Transitions in Gradient Descent', fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График 2: Примеры траекторий
    example_indices = {
        'monotonic': next((i for i, t in enumerate(types) if t == 'monotonic'), None),
        'catapult': next((i for i, t in enumerate(types) if t == 'catapult'), None),
        'periodic': next((i for i, t in enumerate(types) if t == 'periodic'), None),
        'chaotic': next((i for i, t in enumerate(types) if t == 'chaotic'), None),
    }
    
    for ttype, idx in example_indices.items():
        if idx is not None:
            history = quad_results['loss_history'][idx]
            ax2.plot(history, label=f'{ttype} (η={lrs[idx]:.3f})', 
                    color=colors[ttype], linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Example Trajectories', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/figures/phase_transitions.png', dpi=300)
    plt.close()
    
    # Сохранение сводных результатов
    summary = {
        'tanh': {
            'fractal_dimension': float(dim_tanh) if dim_tanh else None,
            'grid_resolution': 512,
        },
        'relu': {
            'fractal_dimension': float(dim_relu) if dim_relu else None,
            'grid_resolution': 512,
        },
        'linear': {
            'fractal_dimension': float(dim_linear) if dim_linear else None,
            'grid_resolution': 512,
        },
        'zoom_sequence': {
            'median_fractal_dimension': float(median_dim) if median_dim else None,
            'num_levels': len(zoom_results),
        }
    }
    
    with open('/mnt/user-data/outputs/data/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*70)
    print("ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ!")
    print("="*70)
    print("\nСводная таблица результатов:")
    print("-" * 70)
    print(f"{'Конфигурация':<20} {'Фрактальная размерность':<30}")
    print("-" * 70)
    print(f"{'tanh':<20} {dim_tanh:.3f if dim_tanh else 'N/A':<30}")
    print(f"{'ReLU':<20} {dim_relu:.3f if dim_relu else 'N/A':<30}")
    print(f"{'Linear':<20} {dim_linear:.3f if dim_linear else 'N/A':<30}")
    print(f"{'Zoom (медиана)':<20} {median_dim:.3f if median_dim else 'N/A':<30}")
    print("-" * 70)
    
    print("\nРезультаты сохранены в:")
    print("  - Графики: /mnt/user-data/outputs/figures/")
    print("  - Данные: /mnt/user-data/outputs/data/")


if __name__ == '__main__':
    main()
