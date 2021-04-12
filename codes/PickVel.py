import numpy as np
from matplotlib import pyplot as plt
from numpy import array
from sklearn import preprocessing
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture import GaussianMixture
from tqdm import trange


class ForesightDataProcess:
	"""
	完成数据处理工作
	__init__:
		需指定道集cdp_e和道号line_e.
		初始化过程会自动从元数据集中提取出该道集与道号对应的数据和人工拾取的速度点.

	open_data:
		根据路径读取原数据. 储存在self.sth, self.sh, self.data里.

	get_wanted_data:
		在原数据中根据道集和道号进行筛选. 储存在wanted_data中.

	get_true_data:
		根据道集和道号筛选出人工拾取的速度点.

	get_pre_cluster_data:
		进一步处理, 得到待聚类的数据.

	clean:
		按照threshold和是否正则化对数据进行清洗.

	reverse_clean:
		对正则化后的数据进行还原.
	"""

	def __init__(self, cdp_e: int = 0, line_e: int = 0):
		sth_path = "data/速度谱/STH_combin_pwr.txt"
		sh_path = "data/速度谱/SH_combin_pwr.txt"
		data_path = "data/速度谱/data_combin_pwr.npy"

		self.sth = list()
		self.sh = list()
		self.data = list()
		self.open_data(sth_path, sh_path, data_path)

		self.pwr_axis_y = list()
		self.pwr_axis_x = list()
		self.cdp_E = 0
		self.line_E = 0

		self.wanted_data_ind = list()
		self.wanted_data = list()
		self.true_data = list()
		self.true_pair = list()

		if cdp_e == 0 or line_e == 0:
			print("cdp道集和道号未指定, 请指定后重新实例化!")
		else:
			self.get_wanted_data(cdp_e, line_e)
			self.get_true_data()

			self.pre_cluster_data = list()
			self.get_pre_cluster_data()

			print('接下来需要指定threshold与是否normalize, 通过clean方法对数据进行清洗!')

			self.threshold = 0
			self.is_normalized = True

			self.scaler = True
			self.normalized_data = list()

	def open_data(self, sth_path, sh_path, data_path):
		f = open(sth_path, 'r')
		self.sth = eval(f.read())
		f.close()

		f = open(sh_path, 'r')
		self.sh = eval(f.read())
		f.close()

		self.data = np.load(data_path)

	def get_wanted_data(self, cdp_e, line_e):
		"""
		:param cdp_e: 期望处理的cdp道集.
		:param line_e: 期望处理的line道号.
		return: 处理好的数据.
		"""
		self.cdp_E = cdp_e
		self.line_E = line_e
		cdp_points = self.sth['cdp']
		line_number = self.sth['Inline3D']
		self.pwr_axis_y = np.array(self.sh['time'] * 1000)
		self.pwr_axis_x = np.array(self.sth['offset'])
		wanted_data_ind = list()

		for i in trange(len(cdp_points)):
			if cdp_points[i] == self.cdp_E and line_number[i] == self.line_E:
				wanted_data_ind.append(i)

		self.wanted_data = self.data[:, wanted_data_ind]
		self.wanted_data_ind = wanted_data_ind
		return self.wanted_data

	def get_true_data(self):
		"""
		return: 与self.cdp_e和self.line_e对应的人工拾取的速度点.
		"""
		path = "data/AI_VelVirtual_Line_Set_PP_Velo_AI.gss"
		l_ind = 0
		c_ind = 0
		stop = 0
		true_data = ''
		line_number = 'LINE' + str(self.line_E)
		cmp_number = 'CMP' + str(self.cdp_E)

		# 在整个拾取数据文件里做搜索
		with open(path, 'r') as f:
			lines = f.readlines()

			for i, line in enumerate(lines):
				if line.lstrip()[:len(line_number)] == line_number:
					l_ind = i
					break
			for i, line in enumerate(lines[l_ind:]):
				if line.lstrip()[:len(cmp_number)] == cmp_number:
					c_ind = l_ind + i
					break
			for i, line in enumerate(lines[c_ind + 1:]):
				if line.lstrip()[:3] == 'CMP':
					stop = c_ind + i + 1
					break
			for i in range(1, stop - c_ind - 1):
				true_data += lines[c_ind + i].lstrip()

		true_data = true_data.replace('\n', '')
		t1 = true_data.split(',')

		true_pair = list()
		for i in t1[:-1]:
			temp = i.split('V')
			true_pair.append([int(temp[0].replace('T', '')), int(temp[1])])

		true_pair = np.array(true_pair)
		true_pair[:, 0] = true_pair[:, 0] / 20
		true_pair[:, 1] = (true_pair[:, 1] - min(self.pwr_axis_x[self.wanted_data_ind])) / 20

		self.true_pair = true_pair
		print("已生成人工拾取的速度点, 储存在self.true_pair中!")
		return true_pair

	def get_pre_cluster_data(self):
		"""
		return: 可以直接进行聚类的数据.
		"""
		temp = self.wanted_data.T.reshape(np.size(self.wanted_data))
		x = len(self.wanted_data[0])
		y = len(self.wanted_data)
		my_cdp = list()

		for i in trange(x):
			for j in range(y):
				my_cdp.append([i + 1, j * 20, temp[i * y + j]])

		self.pre_cluster_data = array(my_cdp)
		return array(my_cdp)

	def clean(self, threshold: float = 0.01, normalized: bool = True):
		"""
		return: 在待聚类数据上清洗后的数据.
		"""
		self.threshold = threshold
		self.is_normalized = normalized
		my_data = self.pre_cluster_data[:, 2]
		my_data = my_data.reshape(-1, 1)
		min_val = my_data.min()
		max_val = my_data.max()
		trans_data = (my_data - min_val) / (max_val - min_val)

		ind = np.where(trans_data >= threshold)
		ind = np.array(ind, dtype=int)[0]

		if normalized:
			scaler = preprocessing.StandardScaler().fit(self.pre_cluster_data[ind, :2])
			normalized_data = scaler.fit_transform(self.pre_cluster_data[ind, :2])
		else:
			scaler = False
			normalized_data = self.pre_cluster_data[ind, :2]

		self.scaler = scaler
		self.normalized_data = normalized_data

		return self.normalized_data, self.scaler

	def reverse_clean(self, data):
		if self.is_normalized:
			reversed_data_x = (data[:, 0] * np.sqrt(self.scaler.var_[0]) + self.scaler.mean_[0]) - 1
			reversed_data_y = (data[:, 1] * np.sqrt(self.scaler.var_[1]) + self.scaler.mean_[1]) / 20
		else:
			reversed_data_x = data[:, 0] - 1
			reversed_data_y = data[:, 1] / 20

		return reversed_data_x, reversed_data_y


class ImpeccableCluster:
	"""
		占个位置
		"""

	def __init__(self, method_name, data):
		self.data = data
		self.centers = list()
		self.labels = list()
		self.method_name = method_name
		self.cluster_number = 0
		all_way = ['K-means', 'DBSCAN', 'GMM_EM', 'GMM_Dirichlet']
		if method_name not in all_way:
			print('请传入聚类方法, 在K-means, DBSCAN, GMM_EM, GMM_Dirichlet中选择. ')
		else:
			flag = all_way.index(method_name)
			if flag == 0:
				print('您选择了K-means方法进行聚类, 请输入聚类中心个数n开始聚类. ')
				n = int(input())
				self.cluster_k_means(n)
			elif flag == 1:
				print('您选择了DBSCAN方法进行聚类, 请输入领域半径Eps和最小个数MSA开始聚类. ')
				eps, msa = float(input('Eps: ')), int(input('MinSample: '))
				self.eps = eps
				self.msa = msa
				self.cluster_dbscan(eps, msa)
			elif flag == 2:
				print(
					'您选择了基于EM算法的高斯混合模型进行聚类, 请输入协方差矩阵类型con_type和聚类中心个数n开始聚类. ')
				n, cov_type = int(input('n: ')), input('Cov_type: ')
				self.n = n
				if not cov_type:
					cov_type = 'full'
				self.cov_type = cov_type
				self.cluster_gmm_em(n, cov_type)
			elif flag == 3:
				print(
					'您选择了基于变分推断的Dirichlet过程高斯混合模型进行聚类, 请输入协方差矩阵类型con_type, 聚类中心个数n, 先验分布类型prior_type, 先验分布参数prior开始聚类. '
				)
				n, cov_type, prior_type, prior = input('n: '), input(
					'Cov_type: '), input('Prior_type: '), input('Prior: ')
				if not n:
					n = 30
				else:
					n = int(n)
				if not cov_type:
					cov_type = 'full'
				if not prior_type:
					prior_type = 'dirichlet_process'
				if not prior:
					prior = 1. / n
				else:
					prior = float(prior)
				self.n = n
				self.cov_type = cov_type
				self.prior_type = prior_type
				self.prior = prior
				self.cluster_gmm_dirichlet(n, cov_type, prior_type, prior)

	def cluster_k_means(self, n=10):
		cluster = KMeans(n)
		cluster.fit(self.data)
		self.centers = cluster.cluster_centers_
		self.labels = cluster.labels_
		self.cluster_number = n

	def cluster_dbscan(self, eps, msa):
		cluster = DBSCAN(eps=eps, min_samples=msa)
		cluster.fit(self.data)
		self.labels = cluster.labels_

		point_list = np.unique(cluster.labels_)
		centers = list()
		for i in point_list[1:]:
			ind = np.where(cluster.labels_ == i)
			x = np.mean(self.data[ind][:, 0])
			y = np.mean(self.data[ind][:, 1])
			centers.append([x, y])
		self.centers = array(centers)

		if -1 in cluster.labels_:
			self.cluster_number = len(np.unique(cluster.labels_)) - 1
		else:
			self.cluster_number = len(np.unique(cluster.labels_))

	def cluster_gmm_em(self, n, cov_type):
		cluster = GaussianMixture(n_components=n, covariance_type=cov_type)
		cluster.fit(self.data)
		if cluster.converged_:
			print('恭喜! 算法收敛!')
			self.centers = cluster.means_
			self.labels = False
			self.cluster_number = n
		else:
			print('不好意思哈, 算法没收敛, 请重试或更改参数后重试!')

	def cluster_gmm_dirichlet(self, n, cov_type, prior_type, prior):
		"""
		'full' (each component has its own general covariance matrix),
		'tied' (all components share the same general covariance matrix),
		'diag' (each component has its own diagonal covariance matrix),
		'spherical' (each component has its own single variance).

		'dirichlet_process' (using the Stick-breaking representation),
		'dirichlet_distribution' (can favor more uniform weights).
		"""
		cluster = BayesianGaussianMixture(n_components=n, covariance_type=cov_type,
		                                  weight_concentration_prior_type=prior_type, weight_concentration_prior=prior,
		                                  max_iter=300)
		cluster.fit(self.data)
		if cluster.converged_:
			print('恭喜! 算法收敛!')
			self.centers = cluster.means_
			self.labels = False
			self.cluster_number = len(np.unique(self.centers[:, 1:2]))
		else:
			print('不好意思哈, 算法没收敛, 请重试或更改参数后重试!')


class ProwessDraw:
	"""

	"""

	def __init__(self, raw_data, clustered_data, raw_point=True, save=False, true_pair=False, result_point=False,
	             centers=False, show=True):
		self.show = show
		self.true_pair = true_pair
		self.raw_point = raw_point
		self.save = save
		self.result_point = result_point
		self.centers = centers

		self.LOW = min(raw_data.pwr_axis_x[raw_data.wanted_data_ind])
		self.UP = max(raw_data.pwr_axis_x[raw_data.wanted_data_ind])
		self.LONG = len(raw_data.wanted_data_ind)

		self.go_draw(raw_data, clustered_data)

	def go_draw(self, raw_data, clustered_data):
		name = ''
		name += clustered_data.method_name

		if self.raw_point:
			plt.imshow(raw_data.wanted_data, cmap='seismic')
			name += '_Raw'

		name += '_Normalized=' + str(raw_data.is_normalized)
		name += 'Threshold=' + str(raw_data.threshold)

		ax = plt.gca()
		ax.xaxis.set_ticks_position('top')
		plt.yticks(np.linspace(0, 349, 5), np.linspace(0, 6980, 5))
		plt.xticks(np.linspace(0, self.LONG - 1, 5), np.linspace(self.LOW, self.UP, 5))

		if self.true_pair:
			plt.plot(raw_data.true_pair[:, 1] - 1, raw_data.true_pair[:, 0], '*', color='red')
			name += '_TruePair'

		if self.result_point:
			reversed_data_x, reversed_data_y = raw_data.reverse_clean(raw_data.normalized_data)
			plt.scatter(reversed_data_x, reversed_data_y, c=clustered_data.labels, s=1)
			name += '_ResultPoint_ClusterNumber=' + str(clustered_data.cluster_number)

		if self.centers:
			x, y = raw_data.reverse_clean(clustered_data.centers)
			plt.scatter(x, y, s=8, color='green')
			name += '_Centers'

		if self.save:
			name += name + '.pdf'
			plt.savefig(name, transparent=True, dpi=600, pad_inches=0, bbox_inches='tight')

		if self.show:
			plt.show()
