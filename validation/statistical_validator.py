#!/usr/bin/env python3
"""
基于统计的验证器 - 当特征相似且数据量少时的替代方案
不依赖深度学习，使用统计方法验证生成图像的合理性
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple
import cv2
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

class StatisticalValidator:
    """基于统计的验证器"""
    
    def __init__(self):
        self.user_statistics = {}
        self.pca = None
        self.user_distributions = {}
    
    def extract_statistical_features(self, image_path: str) -> np.ndarray:
        """提取图像的统计特征"""
        # 加载图像
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.array(Image.open(image_path).convert('L'))
        
        features = []
        
        # 1. 基础统计特征
        features.extend([
            np.mean(img),           # 均值
            np.std(img),            # 标准差
            np.var(img),            # 方差
            stats.skew(img.flatten()),  # 偏度
            stats.kurtosis(img.flatten()),  # 峰度
        ])
        
        # 2. 直方图特征
        hist, _ = np.histogram(img, bins=32, range=(0, 256))
        hist = hist / np.sum(hist)  # 归一化
        features.extend(hist.tolist())
        
        # 3. 纹理特征 (GLCM)
        try:
            from skimage.feature import graycomatrix, graycoprops
            # 计算灰度共生矩阵
            glcm = graycomatrix(img, distances=[1], angles=[0, 45, 90, 135], 
                              levels=256, symmetric=True, normed=True)
            
            # 提取纹理特征
            contrast = graycoprops(glcm, 'contrast').flatten()
            dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
            homogeneity = graycoprops(glcm, 'homogeneity').flatten()
            energy = graycoprops(glcm, 'energy').flatten()
            
            features.extend(contrast.tolist())
            features.extend(dissimilarity.tolist())
            features.extend(homogeneity.tolist())
            features.extend(energy.tolist())
        except ImportError:
            # 如果没有skimage，使用简单的纹理特征
            # 计算梯度特征
            grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            
            features.extend([
                np.mean(np.abs(grad_x)),
                np.std(grad_x),
                np.mean(np.abs(grad_y)),
                np.std(grad_y),
            ])
        
        # 4. 频域特征
        f_transform = np.fft.fft2(img)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        features.extend([
            np.mean(magnitude_spectrum),
            np.std(magnitude_spectrum),
            np.max(magnitude_spectrum),
            np.min(magnitude_spectrum),
        ])
        
        # 5. 形状特征（轮廓）
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                features.append(circularity)
            else:
                features.append(0)
        else:
            features.append(0)
        
        return np.array(features)
    
    def load_user_statistics(self, data_root: str) -> Dict[int, List[np.ndarray]]:
        """加载所有用户的统计特征"""
        print("📊 提取用户统计特征...")
        
        data_path = Path(data_root)
        user_features = {}
        
        for user_dir in data_path.iterdir():
            if user_dir.is_dir() and user_dir.name.startswith('ID_'):
                try:
                    user_id = int(user_dir.name.split('_')[1])
                    images = list(user_dir.glob("*.png")) + list(user_dir.glob("*.jpg"))
                    
                    if images:
                        features = []
                        for img_path in images:
                            try:
                                feat = self.extract_statistical_features(img_path)
                                features.append(feat)
                            except Exception as e:
                                print(f"    警告: 无法处理图像 {img_path}: {e}")
                                continue
                        
                        if features:
                            user_features[user_id] = features
                            print(f"  用户 {user_id}: {len(features)} 张图像的特征")
                
                except ValueError:
                    continue
        
        return user_features
    
    def compute_user_distributions(self, user_features: Dict[int, List[np.ndarray]]):
        """计算每个用户的特征分布"""
        print("📈 计算用户特征分布...")
        
        # 合并所有特征进行PCA降维
        all_features = []
        user_labels = []
        
        for user_id, features in user_features.items():
            all_features.extend(features)
            user_labels.extend([user_id] * len(features))
        
        all_features = np.array(all_features)
        
        # PCA降维到50维（保留主要信息）
        self.pca = PCA(n_components=min(50, all_features.shape[1]))
        all_features_pca = self.pca.fit_transform(all_features)
        
        print(f"  PCA降维: {all_features.shape[1]} -> {all_features_pca.shape[1]}")
        print(f"  解释方差比: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        # 计算每个用户的分布参数
        start_idx = 0
        for user_id, features in user_features.items():
            end_idx = start_idx + len(features)
            user_features_pca = all_features_pca[start_idx:end_idx]
            
            # 计算均值和协方差
            mean = np.mean(user_features_pca, axis=0)
            cov = np.cov(user_features_pca.T)
            
            self.user_distributions[user_id] = {
                'mean': mean,
                'cov': cov,
                'features': user_features_pca
            }
            
            start_idx = end_idx
        
        # 计算用户间的可分离性
        self._analyze_user_separability(all_features_pca, user_labels)
    
    def _analyze_user_separability(self, features: np.ndarray, labels: List[int]):
        """分析用户间的可分离性"""
        print("🔍 分析用户可分离性...")
        
        # 计算轮廓系数
        if len(np.unique(labels)) > 1:
            silhouette_avg = silhouette_score(features, labels)
            print(f"  轮廓系数: {silhouette_avg:.3f} (>0.5为好，>0.7为很好)")
            
            if silhouette_avg < 0.3:
                print("  ⚠️  用户间特征相似度很高，分类可能困难")
            elif silhouette_avg < 0.5:
                print("  ⚠️  用户间特征有一定相似性，需要谨慎验证")
            else:
                print("  ✅ 用户间特征有较好的可分离性")
        
        # 使用t-SNE可视化（可选）
        try:
            if len(features) < 1000:  # 数据量不大时才做t-SNE
                tsne = TSNE(n_components=2, random_state=42)
                features_2d = tsne.fit_transform(features)
                
                plt.figure(figsize=(10, 8))
                unique_labels = np.unique(labels)
                colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
                
                for label, color in zip(unique_labels, colors):
                    mask = np.array(labels) == label
                    plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                              c=[color], label=f'User {label}', alpha=0.6)
                
                plt.title('用户特征分布 (t-SNE)')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                plt.savefig('user_feature_distribution.png', dpi=150, bbox_inches='tight')
                plt.close()
                print("  📊 特征分布图已保存: user_feature_distribution.png")
        except Exception as e:
            print(f"  注意: t-SNE可视化失败: {e}")
    
    def validate_generated_images(self, target_user_id: int, 
                                generated_images_dir: str) -> Dict:
        """验证生成图像的统计合理性"""
        print(f"\n🔍 统计验证生成图像 (用户 {target_user_id})")
        
        if target_user_id not in self.user_distributions:
            return {'error': f'用户 {target_user_id} 的分布信息不存在'}
        
        # 加载生成图像
        gen_dir = Path(generated_images_dir)
        image_files = list(gen_dir.glob("*.png")) + list(gen_dir.glob("*.jpg"))
        
        if not image_files:
            return {'error': 'No generated images found'}
        
        print(f"  找到 {len(image_files)} 张生成图像")
        
        # 提取生成图像的特征
        gen_features = []
        for img_path in image_files:
            try:
                feat = self.extract_statistical_features(img_path)
                gen_features.append(feat)
            except Exception as e:
                print(f"    警告: 无法处理生成图像 {img_path}: {e}")
                continue
        
        if not gen_features:
            return {'error': 'No valid generated images'}
        
        gen_features = np.array(gen_features)
        
        # PCA变换
        gen_features_pca = self.pca.transform(gen_features)
        
        # 获取目标用户的分布
        target_dist = self.user_distributions[target_user_id]
        target_mean = target_dist['mean']
        target_cov = target_dist['cov']
        
        # 计算马氏距离
        try:
            inv_cov = np.linalg.pinv(target_cov)  # 使用伪逆避免奇异矩阵
            mahalanobis_distances = []
            
            for gen_feat in gen_features_pca:
                diff = gen_feat - target_mean
                distance = np.sqrt(diff.T @ inv_cov @ diff)
                mahalanobis_distances.append(distance)
            
            mahalanobis_distances = np.array(mahalanobis_distances)
            
            # 计算目标用户真实图像的马氏距离分布作为参考
            real_distances = []
            for real_feat in target_dist['features']:
                diff = real_feat - target_mean
                distance = np.sqrt(diff.T @ inv_cov @ diff)
                real_distances.append(distance)
            
            real_distances = np.array(real_distances)
            
            # 统计分析
            # 使用Kolmogorov-Smirnov检验比较分布
            ks_statistic, p_value = stats.ks_2samp(mahalanobis_distances, real_distances)
            
            # 计算生成图像在真实分布中的百分位数
            percentiles = [stats.percentileofscore(real_distances, d) for d in mahalanobis_distances]
            
            # 合理性评估：生成图像应该在真实分布的合理范围内
            reasonable_count = sum(1 for p in percentiles if 5 <= p <= 95)  # 在5%-95%范围内
            reasonable_rate = reasonable_count / len(percentiles)
            
            result = {
                'reasonable_rate': reasonable_rate,
                'avg_mahalanobis_distance': np.mean(mahalanobis_distances),
                'real_avg_mahalanobis_distance': np.mean(real_distances),
                'ks_statistic': ks_statistic,
                'ks_p_value': p_value,
                'distribution_similar': p_value > 0.05,  # p>0.05表示分布相似
                'percentiles': percentiles,
                'total_images': len(mahalanobis_distances)
            }
            
            print(f"  📊 验证结果:")
            print(f"    合理性比率: {reasonable_rate:.3f}")
            print(f"    分布相似性: {'是' if result['distribution_similar'] else '否'} (p={p_value:.3f})")
            print(f"    平均马氏距离: 生成={np.mean(mahalanobis_distances):.3f}, 真实={np.mean(real_distances):.3f}")
            
            return result
            
        except Exception as e:
            print(f"    ❌ 马氏距离计算失败: {e}")
            return {'error': f'Statistical analysis failed: {e}'}

# 使用示例
if __name__ == "__main__":
    validator = StatisticalValidator()
    
    # 加载用户统计特征
    user_features = validator.load_user_statistics("data/processed")
    
    # 计算用户分布
    validator.compute_user_distributions(user_features)
    
    # 验证生成图像
    result = validator.validate_generated_images(
        target_user_id=1,
        generated_images_dir="validation_results/generated_images"
    )
