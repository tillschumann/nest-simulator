/*
 * kernels.h
 *
 *  Created on: Jul 20, 2016
 *      Author: schumann
 */

#ifndef KERNELS_H_
#define KERNELS_H_

template <typename T>
struct kernel {
	virtual std::vector<T> operator()(std::vector<T> values)
	{
		return values;
	}
};

template <typename T>
struct kernel_combi : public kernel<T> {
	std::vector< kernel<T>* > kernels_;
	kernel_combi()
	{
		kernel<T>* k = new kernel<T>();
		kernels_.push_back(k);
	}

	~kernel_combi()
	{
		for (int i=0; i<kernels_.size(); i++)
			delete kernels_[i];
	}

	template<typename K>
	void push_back(const std::vector<T>& v)
	{
		K* k = new K(v);
		kernels_.push_back( static_cast<kernel<T>*>(k) );
	}
	std::vector<T> func(std::vector<T> values)
	{
		for (int i=0; i<kernels_.size(); i++)
			values = kernels_[i](values);
		return values;
	}
};

template <typename T>
struct kernel_multi : public kernel<T> {
	std::vector<T> multis_;
	kernel_multi(const std::vector<T>& multis): multis_(multis)
	{}
	std::vector<T> func(std::vector<T> values)
	{
		assert(values.size() == multis_.size());

		for (int i=0; i<values.size(); i++)
			values[i] *= multis_[i];
		return values;
	}
};

template <typename T>
struct kernel_add : public kernel<T> {
	std::vector<T> adds_;
	kernel_add(const std::vector<T>& adds): adds_(adds)
	{}
	std::vector<T> func(std::vector<T> values)
	{
		assert(values.size() == adds_.size());

		for (int i=0; i<values.size(); i++)
			values[i] += adds_[i];
		return values;
	}
};

template <typename T>
struct kernel_csaba : public kernel<T> {
	T lower;
	T upper;
	kernel_csaba(const std::vector<T>& boundaries)
	{
		assert(boundaries.size() = 2);
		lower = boundaries[0];
		upper = boundaries[1];
	}
	std::vector<T> func(std::vector<T> values)
	{
		assert(values.size() == 5);

		const double Vprop = 0.8433734 * 1000.0;
		const double distance = values[0] * Vprop;
		if (distance <= lower)
			values[1] *= 2.0;
		else if (distance < upper)
			values[1] *= -2.0;

		return values;
	}
};

#endif /* KERNELS_H_ */
