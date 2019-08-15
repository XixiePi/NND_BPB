import Dataset as ds
import DataIterator as di

dataset = ds.Dataset(name="test_dataset")

dataset.set_data()

data_iterator = di.DataIterator(dataset=dataset)

data_iterator.iteration_starter(iteration_method="single")

for i in range(10):
    [p, t] = data_iterator.next()

    # NOTE: the compare between p and "end" should use is.
    if p is "end":
        break
    print("Looping item:", data_iterator.iter_num, " idx:", data_iterator.idx, p, t)
