import dataset

featData = dataset.InputData(data_folder='./samples')

feat, image = next(iter(featData))
print(feat.shape, image.shape)