from msilib.schema import Font
import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.interpolate import CubicSpline

# 可复现随机数
rng = np.random.RandomState(0)

# === 定义真实函数 ===
def f(x):
    return np.sin(3.0 * x) * 0.5 + 0.5 * np.cos(5.0 * x)

# 采样点
X_samples = np.array([0.15, 0.48, 0.9, 1.25, 1.6]).reshape(-1, 1)
y_samples = f(X_samples).ravel()

# Gaussian Process 拟合
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=0.3, length_scale_bounds=(1e-2, 1.0))
gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=5, random_state=0)
gpr.fit(X_samples, y_samples)

X = np.linspace(0.0, 2.0, 400).reshape(-1, 1)
y_true = f(X).ravel()
y_mean, y_std = gpr.predict(X, return_std=True)
y_mean = y_mean.ravel()
y_std = y_std.ravel()

# 三次样条经过采样点
cs = CubicSpline(X_samples.ravel(), y_samples, bc_type='natural')
y_spline = cs(X.ravel())

# 绘制
#plt.style.use('seaborn-v0_8')
plt.rcParams["font.family"] = "Times New Roman"

fig, ax = plt.subplots(figsize=(7.5, 4.5))

ax.plot(X, y_true, linestyle='--', color='k', label='True function $f(x)$', linewidth=1.4)
ax.plot(X, y_mean, color='tab:blue', label='GP posterior mean', linewidth=2)
ax.fill_between(X.ravel(), y_mean - 1.96 * y_std, y_mean + 1.96 * y_std,
                color='tab:blue', alpha=0.25, label='GP 95% CI')
ax.scatter(X_samples.ravel(), y_samples, color='tab:red', s=60, zorder=10, label='Observed points')
ax.plot(X, y_spline, color='orange', linestyle='-', linewidth=1.6, alpha=0.9, label='Spline through samples')

# for xs, ys in zip(X_samples.ravel(), y_samples):
#     ax.annotate(f'({xs:.2f}, {ys:.2f})', xy=(xs, ys), xytext=(5, 6), textcoords='offset points', fontsize=8)

ax.set_xlabel('x')
ax.set_ylabel('f(x) / GP')
ax.set_title('Illustration of Bayesian Optimization',fontsize=12)
ax.legend(loc='best',frameon=False, fontsize=9)
ax.set_xlim(0.0, 2.0)
ymin = min(y_true.min(), y_mean.min(), y_spline.min()) - 0.3
ymax = max(y_true.max(), y_mean.max(), y_spline.max()) + 0.3
ax.set_ylim(ymin, ymax)

plt.tight_layout()
plt.savefig('Result/bayes_opt_illustration.png', dpi=300)
plt.show()