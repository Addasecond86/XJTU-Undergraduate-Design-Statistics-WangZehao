import numpy as np
from matplotlib import pyplot as plt
from numpy import array
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.optimize import curve_fit
from sklearn import preprocessing
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from tqdm import trange


class ForesightDataProcess:
	"""
	This class is used to do the processing of the raw data.

	__init__:
			Read the data, and also read the manual analysis results.
			You need to specify the cdp number 'cdp_e' and the gather number 'line_e'.

	open_data:
			Function for reading data.

	get_wanted_data:
			Function to filter the raw data. Get the data to be processed, and store it in 'wanted_data'.

	get_true_data:
			Function for reading manual analysis results.

	get_pre_cluster_data:
			Normalize functions.

	clean:
			Data Cleaning.

	reverse_clean:
			Function to restore the cleaned data.
	"""

	def __init__(self, cdp_e: int = 0, line_e: int = 0,
	             sth_path="data/速度谱/STH_combin_pwr.txt",
	             sh_path="data/速度谱/SH_combin_pwr.txt",
	             data_path="data/速度谱/data_combin_pwr.npy"):

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
			print("Cdp number or Gather number is not specified, please specify and re-instantiate! ")
		else:
			self.get_wanted_data(cdp_e, line_e)
			self.get_true_data()

			self.pre_cluster_data = list()
			self.get_pre_cluster_data()

			print(
				'Next, you need to specify the threshold and whether to normalize, and clean the data by the clean method! ')

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
		return: Processed data.
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
		return: Manual analysis of speed points.
		"""
		path = "data/AI_VelVirtual_Line_Set_PP_Velo_AI.gss"
		l_ind = 0
		c_ind = 0
		stop = 0
		true_data = ''
		line_number = 'LINE' + str(self.line_E)
		cmp_number = 'CMP' + str(self.cdp_E)

		# Searching through the entire pickup data file.
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
		true_pair[:, 1] = (true_pair[:, 1] -
		                   min(self.pwr_axis_x[self.wanted_data_ind])) / 20

		self.true_pair = true_pair
		print("The velocity points of the manual analysis have been generated, and stored in self.true_pair! ")
		return true_pair

	def get_pre_cluster_data(self):
		"""
		return: The data can be directly clustered.
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
		return: Cleaned data.
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
			scaler = preprocessing.StandardScaler().fit(
				self.pre_cluster_data[ind, :2])
			normalized_data = scaler.fit_transform(
				self.pre_cluster_data[ind, :2])
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
	This class is used to implement clustering.

	__init__:
			The clustering is done directly in this function, and all other functions are auxiliary functions.
	"""

	def __init__(self, method_name, data):
		self.data = data
		self.centers = list()
		self.labels = list()
		self.method_name = method_name
		self.cluster_number = 0

		all_way = ['K-means', 'DBSCAN', 'GMM_EM', 'GMM_Dirichlet']

		if method_name not in all_way:
			print('Please pass in the clustering method, choose among "K-means", "DBSCAN", "GMM_EM", "GMM_Dirichlet". ')
		else:
			flag = all_way.index(method_name)

			if flag == 0:
				print(
					'You have selected the K-means method for clustering, please enter the number of cluster centers n to start clustering. ')
				n = int(input())

				self.cluster_k_means(n)
			elif flag == 1:
				print(
					'You have selected the DBSCAN method for clustering, please enter the domain radius Eps and the minimum number of MSA to start clustering. ')
				eps, msa = float(input('Eps: ')), int(input('MinSample: '))

				self.eps = eps
				self.msa = msa
				self.cluster_dbscan(eps, msa)
			elif flag == 2:
				print(
					'You have selected a Gaussian mixture model based on EM algorithm for clustering, please enter the covariance matrix type con_type and the number of clustering centers n to start clustering. ')
				n, cov_type = int(input('n: ')), input('Cov_type: ')

				self.n = n

				if not cov_type:
					cov_type = 'full'

				self.cov_type = cov_type
				self.cluster_gmm_em(n, cov_type)
			elif flag == 3:
				print(
					'You have selected the Dirichlet process Gaussian mixture model based on variational inference for clustering, please enter the covariance matrix type con_type, the number of clustering centers n, the prior distribution type prior_type, and the prior distribution parameter prior to start clustering. '
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
			print('Congratulations! The algorithm converges!')
			self.centers = cluster.means_
			self.labels = False
			self.cluster_number = n
		else:
			print('Sorry, the algorithm did not converge, please try again or change the parameters and try again!')

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
			print('Congratulations! The algorithm converges!')
			self.rm_repeat_point(cluster.means_)
			self.labels = False
		else:
			print('Sorry, the algorithm did not converge, please try again or change the parameters and try again!')

	def rm_repeat_point(self, target):
		"""
		Removing duplicate points.
		"""
		x = target[:, 0]
		y = target[:, 1]

		order = np.argsort(x)
		x.sort()

		sorted_data = list()
		for i, j in enumerate(order):
			sorted_data.append((x[i], y[j]))

		sorted_data = array(sorted_data)

		del_target = list()
		for i in range(len(sorted_data) - 1):
			dis = np.abs(sorted_data[i][0] - sorted_data[i][1]) ** 2 + np.abs(
				sorted_data[i + 1][0] - sorted_data[i + 1][1]) ** 2
			if dis <= 0.001:
				del_target.append(i)

		for i in reversed(del_target):
			sorted_data = np.delete(sorted_data, i, axis=0)

		self.centers = sorted_data
		self.cluster_number = len(sorted_data)

	def rm_noise(self):
		del_target = list()
		processed_data = self.centers

		def square_fitting(x1, y1):
			def fun(x, a, b, c):
				return a * x + b * x ** 2 + c

			coefficient, _ = curve_fit(fun, x1, y1)
			return coefficient

		coefficient1 = square_fitting(
			processed_data[:, 0], processed_data[:, 1])

		def r_fun(x):
			return coefficient1[0] * x + coefficient1[1] * x ** 2 + coefficient1[2]

		for i in range(len(processed_data)):
			if np.abs(r_fun(processed_data[i, 0]) - processed_data[i, 1]) > 0.5:
				del_target.append(i)

		for i in reversed(del_target):
			processed_data = np.delete(processed_data, i, axis=0)

		self.centers = processed_data
		self.cluster_number = len(processed_data)
		return processed_data


class ProwessDraw:
	"""
	The class used to integrate drawing.
	"""

	def __init__(self, raw_data, clustered_data, fitting_function=None, raw_point=True, save=False, true_pair=False,
	             result_point=False,
	             centers=False, show=True, fitting=False, brief=True, self_centers=None):
		self.show = show
		self.true_pair = true_pair
		self.raw_point = raw_point
		self.save = save
		self.result_point = result_point
		self.centers = centers
		self.fitting = fitting
		self.brief = brief
		self.fitting_function = fitting_function
		self.self_centers = self_centers

		self.LOW = min(raw_data.pwr_axis_x[raw_data.wanted_data_ind])
		self.UP = max(raw_data.pwr_axis_x[raw_data.wanted_data_ind])
		self.LONG = len(raw_data.wanted_data_ind)

		self.go_draw(raw_data, clustered_data)

		if clustered_data.method_name == 'GMM_EM' or clustered_data.method_name == 'GMM_Dirichlet':
			self.raw_point = False

	def go_draw(self, raw_data, clustered_data):
		plt.figure(figsize=(5, 6))
		name = ''
		name += clustered_data.method_name

		if self.raw_point:
			plt.imshow(raw_data.wanted_data, cmap='seismic')
			if not self.brief:
				name += '_Raw'

		if not self.brief:
			name += '_Normalized=' + str(raw_data.is_normalized)
			name += '_Threshold=' + str(raw_data.threshold)

		ax = plt.gca()
		ax.xaxis.set_ticks_position('top')
		plt.yticks(np.linspace(0, 349, 5), np.linspace(0, 6980, 5))
		plt.xticks(np.linspace(0, self.LONG - 1, 5),
		           np.linspace(self.LOW, self.UP, 5))

		if self.fitting:
			if self.fitting_function is None:
				return 'There is no function passed into the fitted out! '
			x_new = np.linspace(0, self.LONG - 1, 30)
			y_new = self.fitting_function.result_function(x_new)
			del_target = list()
			for i in range(len(y_new)):
				if y_new[i] > 349 or y_new[i] < 0:
					del_target.append(i)

			for i in reversed(del_target):
				x_new = np.delete(x_new, i, axis=0)
				y_new = np.delete(y_new, i, axis=0)

			plt.plot(x_new, y_new, '--')
			name += '_FittingMethod=' + str(self.fitting_function.method)

		if self.true_pair:
			plt.plot(
				raw_data.true_pair[:, 1] - 1, raw_data.true_pair[:, 0], '*', color='red', alpha=0.25)
			name += '_TruePair'

		if self.result_point:
			reversed_data_x, reversed_data_y = raw_data.reverse_clean(
				raw_data.normalized_data)
			plt.scatter(reversed_data_x, reversed_data_y,
			            c=clustered_data.labels, s=1)
			name += '_ResultPoint_ClusterNumber=' + str(clustered_data.cluster_number)

		if self.centers:
			if self.self_centers is None:
				x, y = raw_data.reverse_clean(clustered_data.centers)
			else:
				x, y = self.self_centers[:, 0], self.self_centers[:, 1]
			plt.scatter(x, y, s=10, color='yellow')
			name += '_Centers'

		all_way = ['K-means', 'DBSCAN', 'GMM_EM', 'GMM_Dirichlet']
		flag = all_way.index(clustered_data.method_name)
		if flag == 0:
			name += '_n=' + str(clustered_data.cluster_number)
		elif flag == 1:
			name += '_Eps=' + str(clustered_data.eps) + '_MSA=' + str(clustered_data.msa)
		elif flag == 2:
			name += '_n=' + str(clustered_data.n) + '_CovType=' + str(clustered_data.cov_type)
		elif flag == 3:
			name += '_Real_n=' + str(clustered_data.n) + '_n=' + str(clustered_data.cluster_number) + '_CovType=' + str(
				clustered_data.cov_type) + '_PriorType=' + str(
				clustered_data.prior_type) + '_Prior=' + str(np.round(clustered_data.prior, 2))

		if self.save:
			name += '.pdf'
			plt.savefig(name, transparent=True, dpi=600,
			            pad_inches=0, bbox_inches='tight')

		if self.show:
			plt.show()


class ChumFitting:
	"""
	The class used to fit the clustering results.
	"""

	def __init__(self, raw_data):
		self.data = raw_data
		self.result_function = list()
		self.x, self.y = self.data[:][0], self.data[:][1]
		print('Now to curve-fit the clustering results, enter 1, 2 or 3. \n'
		      '1 denotes fitting with a quadratic curve; \n'
		      '2 denotes fitting by the ordinary interpolation method; \n'
		      '3 indicates fitting by spline interpolation. ')

		m = int(input())

		if m == 1:
			self.method = 'Square_Curve'
			print('You have selected the quadratic curve, so start fitting it!')
			print('Please enter 1 or 2, if one of the numbers is wrong, change the other!')
			n = int(input('n='))

			if n == 1:
				coefficient = self.square_fitting1()

				def r_fun(x):
					return coefficient[0] * x + coefficient[1] * x ** 2 + coefficient[2]

				self.result_function = r_fun
			elif n == 2:
				coefficient = self.square_fitting2()

				def r_fun(x):
					return coefficient[0] * x + coefficient[1] * x ** 2

				self.result_function = r_fun

		elif m == 2:
			self.method = 'Interpolation'
			print('You have chosen the interpolation method, so start fitting!')

			self.result_function = interp1d(
				self.x, self.y, fill_value="extrapolate")

		elif m == 3:
			self.method = 'Interpolation'
			print("You have chosen the spline interpolation method, so let's start the fitting!")
			self.result_function = UnivariateSpline(self.x, self.y)

	def square_fitting1(self):
		def fun1(x, a, b, c):
			return a * x + b * x ** 2 + c

		coefficient, _ = curve_fit(fun1, self.x, self.y)
		return coefficient

	def square_fitting2(self):
		def fun2(x, a, b):
			return a * x + b * x ** 2

		coefficient, _ = curve_fit(fun2, self.x, self.y)
		return coefficient
