import os
from pickle import FALSE
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, Memory
from sklearn.decomposition import TruncatedSVD
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree
from scipy.interpolate import RBFInterpolator
# from matplotlib.tri import Triangulation
# import matplotlib.pyplot as plt
import threading
from tqdm import tqdm
from lib.Counter_run_time import CallingCounter

class TqdmParallel(Parallel):
    def __init__(self, use_tqdm=True, total=0, **kwargs):
        self._use_tqdm = use_tqdm
        self._lock = threading.Lock()
        self._pbar = tqdm(total=total) if use_tqdm else None
        super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        self._i = 0
        return super().__call__(*args, **kwargs)

    def print_progress(self):
        with self._lock:
            self._i += 1
            if self._pbar:
                self._pbar.update(1)

    def _backend_callback(self, *args, **kwargs):
        self.print_progress()
        return super()._backend_callback(*args, **kwargs)

    def __del__(self):
        if self._pbar:
            self._pbar.close()


class LocalRBFVelocityInterpolator:
    def __init__(self, BC, V_snapshots, svd, kernel='thin_plate_spline', smoothing=1e-6, neighbors=30):
        self.newBC = BC[:,[0,2,3]]
        self.bc_min = self.newBC.min(axis=0)
        self.bc_max = self.newBC.max(axis=0)
        self.BC_norm = (self.newBC - self.bc_min) / (self.bc_max - self.bc_min + 1e-12)

        # —— 速度做“去均值”的 SVD（强烈推荐）——
        self.V_mean = V_snapshots.mean(axis=0)
        V_fluc = V_snapshots - self.V_mean
        self.svd = svd
        self.coeffs = self.svd.fit_transform(V_fluc)  # 用去均值后的数据拟合

        # 自动设置 epsilon（可选）：用中位数距离作为尺度
        from scipy.spatial import distance_matrix
        D = distance_matrix(self.BC_norm, self.BC_norm)
        med = np.median(D[D>0])
        self.epsilon = med if np.isfinite(med) and med>0 else 0.2

        # 直接用RBFInterpolator做多输出插值（目标是 coeffs）
        self.rbf = RBFInterpolator(
            self.BC_norm,
            self.coeffs,
            kernel=kernel, epsilon=self.epsilon,
            smoothing=smoothing,
            neighbors=neighbors  # 让它内部用平滑的局部近邻
        )

    def normalize(self, x):
        return (x - self.bc_min) / (self.bc_max - self.bc_min + 1e-12)

    def predict(self, new_bc):
        x = np.asarray(new_bc)[[0,2,3]]
        # 防止轻微出界的外推不稳：clip回训练范围（也可只发警告）
        x = np.minimum(np.maximum(x, self.bc_min), self.bc_max)
        x_norm = self.normalize(x)

        coeffs_pred = self.rbf(x_norm[None, :])[0]
        V_flat = self.svd.inverse_transform(coeffs_pred[None, :])[0] + self.V_mean
        return V_flat, coeffs_pred

# === L 矩阵插值 ===
class LocalRBFGalerkinMatrixInterpolator:
    def __init__(self, BC, L_list, r, 
                 kernel='multiquadric', smoothing=1e-6, neighbors=30):
        # 只取 [0,2,3] 维度
        self.newBC = BC
        self.bc_min = self.newBC.min(axis=0)
        self.bc_max = self.newBC.max(axis=0)
        self.BC_norm = (self.newBC - self.bc_min) / (self.bc_max - self.bc_min + 1e-12)

        self.r = r
        # 展平成二维
        self.L_list = L_list.reshape(len(BC), -1)

        # 自动设置 epsilon（与速度插值一致）
        from scipy.spatial import distance_matrix
        D = distance_matrix(self.BC_norm, self.BC_norm)
        med = np.median(D[D > 0])
        self.epsilon = med if np.isfinite(med) and med > 0 else 0.2

        # RBF 拟合 L 矩阵系数
        self.rbf = RBFInterpolator(
            self.BC_norm,
            self.L_list,
            kernel=kernel, epsilon=self.epsilon,
            smoothing=smoothing,
            neighbors=neighbors
        )

    def normalize(self, x):
        return (x - self.bc_min) / (self.bc_max - self.bc_min + 1e-12)

    def predict(self, new_bc):
        x = np.asarray(new_bc)
        # 裁剪到训练范围
        x = np.minimum(np.maximum(x, self.bc_min), self.bc_max)
        x_norm = self.normalize(x)

        L_flat = self.rbf(x_norm[None, :])[0]
        return L_flat.reshape((self.r, self.r))

# class LocalRBFGalerkinMatrixInterpolator:
#     def __init__(self, BC, L_list, r, k=10, kernel='multiquadric', epsilon=1.5):
#         self.BC = np.asarray(BC)
#         self.L_list = L_list.reshape(len(BC), -1)  # flatten L matrices
#         self.r = r
#         self.k = k
#         self.kernel = kernel
#         self.epsilon = epsilon
#         self.tree = cKDTree(self.BC)

#     def predict(self, new_bc):
#         dists, idxs = self.tree.query(new_bc, k=self.k)
#         local_BC = self.BC[idxs]
#         local_L = self.L_list[idxs]
#         rbf = RBFInterpolator(local_BC, local_L, kernel=self.kernel, epsilon=self.epsilon)
#         L_flat = rbf([new_bc])[0]
#         return L_flat.reshape((self.r, self.r))

class predictFlow:
    # === POD ===
    def __init__(self,**kwargs):
        #self.data_dir = "D:/NextPaper/code/AutoCFD/Workdata/Fluent_Python/NEWUSEDATA"
        self.coords = kwargs['coords']
        self.T_snaps=kwargs['T_snaps']
        self.Mass_snaps=kwargs['Mass_snaps']
        self.V_snaps=kwargs['V_snaps']
        self.Vm_snaps=kwargs['Vm_snaps']
        self.BC = kwargs['BC']
        self.velocity = kwargs['velocity']
        self.temperature = kwargs['temperature']
        self.a1 = kwargs['a1']
        self.a2 = kwargs['a2']
        self.force_recompute_L = False
        self.savedir = kwargs['work_path']
    
    def compute_POD(self, T_snapshots, r):
        mean_T = np.mean(T_snapshots, axis=0)
        fluctuations = T_snapshots - mean_T
        svd = TruncatedSVD(n_components=r)
        coeffs = svd.fit_transform(fluctuations)
        modes = svd.components_
        explained_ratios = svd.explained_variance_ratio_
        # 总共保留了多少能量
        energy_retention = np.sum(explained_ratios)
        print(f"当前选用的 {r} 个模态共保留浓度能量比例: {energy_retention:.4f}")
        print("POD temperature modes computed.")
        return modes, coeffs, mean_T, svd

    # === 邻域 & 导数计算 ===
    def compute_neighbor_data(self, coords, k=15):
        tree = cKDTree(coords)
        dists, idxs = tree.query(coords, k=k)
        return idxs

    def fast_gradient(self, phi_modes, coords, neighbors_idx):
        r, N = phi_modes.shape
        gradients = np.zeros((r, N, 3))

        for i in range(N):
            nbr_ids = neighbors_idx[i]
            xi = coords[i]
            X = coords[nbr_ids] - xi  # (k, 3)
            distances = np.linalg.norm(X, axis=1) + 1e-12  # 防止除以0
            X_normalized = X / distances[:, None]

            weights = np.exp(-distances**2 / (np.mean(distances)**2 + 1e-12))
            W = np.diag(weights)

            A = X_normalized
            AT_W = A.T @ W
            H = AT_W @ A + 1e-8 * np.eye(3)  # 正则化稳定求逆

            try:
                H_inv = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                H_inv = np.linalg.pinv(H)

            for j in range(r):
                dphi = phi_modes[j, nbr_ids] - phi_modes[j, i]
                gradients[j, i] = H_inv @ AT_W @ dphi

        print("修正后梯度 max abs:", np.max(np.abs(gradients)))
        return gradients

    def fast_laplacian(self, phi_modes, coords, neighbors_idx):
        r, N = phi_modes.shape
        laplacian = np.zeros((r, N))

        for i in range(N):
            nbr_ids = neighbors_idx[i]
            xi = coords[i]
            X = coords[nbr_ids] - xi  # (k, 3)

            # 特征矩阵 A: 多项式项
            x, y, z = X[:, 0], X[:, 1], X[:, 2]
            A = np.stack([
                np.ones_like(x), x, y, z,
                x*x, y*y, z*z,
                x*y, y*z, z*x
            ], axis=1)

            # 权重矩阵（基于距离）
            distances = np.linalg.norm(X, axis=1)
            weights = np.exp(-distances**2 / (np.mean(distances)**2 + 1e-12))
            W = np.diag(weights)

            AT_W = A.T @ W
            H = AT_W @ A + 1e-8 * np.eye(A.shape[1])
            try:
                H_inv = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                H_inv = np.linalg.pinv(H)

            for j in range(r):
                dphi = phi_modes[j, nbr_ids] - phi_modes[j, i]
                coeffs = H_inv @ AT_W @ dphi

                # 提取二阶导系数
                laplacian[j, i] = 2 * (coeffs[4] + coeffs[5] + coeffs[6])  # 2*(φ_xx + φ_yy + φ_zz)

        print("修正后Laplacian max abs:", np.max(np.abs(laplacian)))
        return laplacian

    # === 缓存梯度与拉普拉斯 ===
    def get_grad_lap_path(self, r, N):
        return f"lib/grad_lap_r{r}_N{N}_co2_NEWDATAnewCombine.npz"

    def save_gradient_laplacian(self, gradients, laplacians, r, N):
        path = self.get_grad_lap_path(r, N)
        np.savez_compressed(path, gradients=gradients, laplacians=laplacians)
        print(f"梯度和拉普拉斯缓存已保存至: {path}")

    def load_gradient_laplacian(self, r, N):
        path = self.get_grad_lap_path(r, N)
        if os.path.exists(path):
            data = np.load(path)
            print(f"从 {path} 加载梯度和拉普拉斯缓存")
            return data['gradients'], data['laplacians']
        return None, None

    def ensure_gradient_laplacian_cached(self, phi_modes, coords, neighbors_idx):
        r, N = phi_modes.shape
        gradients, laplacians = self.load_gradient_laplacian(r, N)
        if gradients is None or laplacians is None:
            print("首次计算梯度与拉普拉斯（主线程）...")
            gradients = self.fast_gradient(phi_modes, coords, neighbors_idx)
            laplacians = self.fast_laplacian(phi_modes, coords, neighbors_idx)
            self.save_gradient_laplacian(gradients, laplacians, r, N)
        else:
            print("已检测到梯度与拉普拉斯缓存")
        return gradients, laplacians

    # === Galerkin 矩阵 ===
    def build_galerkin_matrix(self, phi_modes, velocity_field, gradients, laplacians, alpha):
        r, N = phi_modes.shape
        L = np.zeros((r, r))
        for i in range(r):
            adv = np.sum(velocity_field * gradients[i], axis=1)
            for k in range(r):
                conv_term = np.dot(phi_modes[k], adv)
                diff_term = np.dot(phi_modes[k], laplacians[i])
                L[k, i] = -conv_term + alpha * diff_term
        return L

    # === 缓存系统 ===
    L_CACHE_PATH = "lib/precomputed_L_matrices_ori_r_18_co2__NEWDATAnewCombine.npz"
    L_T_CACHE_PATH = 'lib/precomputed_L_T_matrices_ori_r_18_co2__NEWDATAnewCombine.npz'
    NEIGHBOR_CACHE_PATH = "lib/neighbors_idx_ori_r_18_co2_NEWDATAnewCombine.npz"

    def save_L_matrices(self, BC, L_list, path=L_CACHE_PATH):
        np.savez(path, BC=BC, L_list=L_list)
        print(f"L矩阵和BC已保存至: {path}")

    def save_L_T_matrices(self, BC, L_list, path=L_T_CACHE_PATH):
        np.savez(path, BC=BC, L_list=L_list)
        print(f"L矩阵和BC已保存至: {path}")

    def load_L_matrices(self, path=L_CACHE_PATH):
        if not os.path.exists(path): return None, None
        data = np.load(path)
        print(f"从 {path} 加载 L 矩阵和BC")
        return data['BC'], data['L_list']

    def load_L_T_matrices(self, path=L_T_CACHE_PATH):
        if not os.path.exists(path): return None, None
        data = np.load(path)
        print(f"从 {path} 加载 L_T 矩阵和BC")
        return data['BC'], data['L_list']

    def save_neighbors_idx(self, neighbors_idx, path=NEIGHBOR_CACHE_PATH):
        np.savez_compressed(path, neighbors_idx=neighbors_idx)
        print(f"邻居索引已保存至: {path}")

    def load_neighbors_idx(self, path=NEIGHBOR_CACHE_PATH):
        if os.path.exists(path):
            data = np.load(path)
            print(f"从 {path} 加载邻居索引")
            return data['neighbors_idx']
        return None

    def precompute_L_matrices_parallel(self, modes_T, coords, V_snapshots, alpha, neighbors_idx, n_jobs=31):
        r, N = modes_T.shape
        gradients, laplacians = self.ensure_gradient_laplacian_cached(modes_T, coords, neighbors_idx)

        total_jobs = len(V_snapshots)
        parallel = TqdmParallel(n_jobs=n_jobs, total=total_jobs, backend="threading")
        results = parallel(
            delayed(self.build_galerkin_matrix)(modes_T, V_snapshots[i], gradients, laplacians, alpha)
            for i in range(total_jobs)
        )
        return np.array(results)

    # === 预测温度 ===
    @CallingCounter
    def predict_temperature(self,new_bc, modes_T, coeffs_T, mean_T,
                        modes_Mass, coeffs_Mass, mean_Mass,
                        velocity_interp, galerkin_interp, galerkin_interp_T,
                        coords, rbf_mass_coeff, rbf_temp_coeff):

        # --- 如果 velocity_interp 用合成速度，这里不用改 ---
        V_f, _ = velocity_interp.predict(new_bc)

        V_field = V_f**2

        # --- 用 RBF 插值得到初始系数（代替 mean） ---
        a0 = rbf_mass_coeff([[new_bc[0],new_bc[2],new_bc[3]]])[0]
        a0_T = rbf_temp_coeff([new_bc])[0]

        # --- Galerkin ODE 演化 ---
        L = galerkin_interp.predict([new_bc[0],new_bc[2],new_bc[3]])
        sol = solve_ivp(lambda t, a: L @ a, (0, 1), a0, t_eval=np.linspace(0, 1, 1000))
        a_final = sol.y[:, -1]
        Mass_pred = mean_Mass + a_final @ modes_Mass

        L_T = galerkin_interp_T.predict(new_bc)
        sol_T = solve_ivp(lambda t, a: L_T @ a, (0, 1), a0_T, t_eval=np.linspace(0, 1, 1000))
        a_T_final = sol_T.y[:, -1]
        T_pred = mean_T + a_T_final @ modes_T

        # --- 保存预测结果 ---
        df = pd.DataFrame({
            "Points:0": coords[:,0], "Points:1": coords[:,1], "Points:2": coords[:,2],
            "Velocity": V_field,  # 这里还是你原来的一维合成速度
            "Temperature": T_pred,
            "Mass_fraction_of_co2": Mass_pred**2
        })
        
        filename = os.path.join(self.savedir, str(self.predict_temperature.count)+'_'+"predicted_snapshot_co2_"+str(round(new_bc[0],2))+'_'+str(round(new_bc[1],2))+'_'+str(round(new_bc[2],2))+'_'+str(round(new_bc[3],2))+".csv")
        df.to_csv(filename, index=False)
        return filename

    # === 主程序 ===
    def main(self):
        #data_dir = self.data_dir
        coords = self.coords
        T_snaps = self.T_snaps
        Mass_snaps = self.Mass_snaps
        V_snaps = self.V_snaps
        Vm_snaps = self.Vm_snaps
        BC = self.BC

        force_recompute_L = self.force_recompute_L

        
        r, rt, alpha = 24, 18, 0.0034

        modes_T, coeffs_T, mean_T, _ = self.compute_POD(T_snaps, rt)
        modes_Mass, coeffs_Mass, mean_Mass, _ = self.compute_POD(Mass_snaps, rt)

        svd_V = TruncatedSVD(n_components=r)
        svd_V.fit(Vm_snaps)
        explained_ratios = svd_V.explained_variance_ratio_
        energy_retention = np.sum(explained_ratios)
        print(f"当前选用的 {r} 个模态共保留速度能量比例: {energy_retention:.4f}")
        print('Velocity POD RBF interpolator trained.')
        local_rbf_interp = LocalRBFVelocityInterpolator(BC, Vm_snaps, svd_V)

        neighbors_idx = self.load_neighbors_idx()
        if neighbors_idx is None:
            print("计算邻居索引...")
            neighbors_idx = self.compute_neighbor_data(coords, k=7)
            self.save_neighbors_idx(neighbors_idx)

        loaded_BC, loaded_L_list = self.load_L_matrices()
        if loaded_BC is not None and loaded_L_list is not None and not force_recompute_L:
            print("使用缓存 L 矩阵。")
            L_list = loaded_L_list
        else:
            print("计算 L 矩阵中...")
            L_list = self.precompute_L_matrices_parallel(modes_Mass, coords, V_snaps, alpha, neighbors_idx)

        loaded_BC, loaded_L_T_list = self.load_L_T_matrices()   
        if loaded_BC is not None and loaded_L_T_list is not None and not force_recompute_L:
            print("使用缓存 L_T 矩阵。")
            L_T_list = loaded_L_T_list
        else:
            print("计算 L_T 矩阵中...")
            L_T_list = self.precompute_L_matrices_parallel(modes_T, coords, V_snaps, alpha, neighbors_idx)
            self.save_L_T_matrices(BC, L_T_list)

        interpolator_L = LocalRBFGalerkinMatrixInterpolator(BC[:,[0,2,3]], L_list, rt)
        interpolator_L_T = LocalRBFGalerkinMatrixInterpolator(BC, L_T_list, rt)
        
        BC_sub = BC[:, [0, 2, 3]]

        # 去掉重复行，并同步去掉 coeffs_Mass 的对应行
        BC_unique, unique_idx = np.unique(BC_sub, axis=0, return_index=True)
        coeffs_Mass_unique = coeffs_Mass[unique_idx]

        # 重新构造 RBF 插值器
        rbf_mass_coeff = RBFInterpolator(BC_unique, coeffs_Mass_unique, kernel="multiquadric", epsilon=1.5)
        #rbf_mass_coeff = RBFInterpolator(BC[:,[0,2,3]], coeffs_Mass, kernel="multiquadric", epsilon=1.5)
        rbf_temp_coeff = RBFInterpolator(BC, coeffs_T, kernel="multiquadric", epsilon=1.5)

        new_bc = [self.velocity,self.temperature,self.a1,self.a2]
        filename = self.predict_temperature(new_bc, modes_T, coeffs_T, mean_T,
                    modes_Mass, coeffs_Mass, mean_Mass,
                    local_rbf_interp, interpolator_L, interpolator_L_T,
                    coords, rbf_mass_coeff, rbf_temp_coeff)
        
        return filename
