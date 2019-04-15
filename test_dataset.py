import dataset

featData = dataset.InputData(data_folder='./samples')
print(len(featData))

feat, image, alpha = next(iter(featData))
print(feat.shape, image.shape, alpha.shape)