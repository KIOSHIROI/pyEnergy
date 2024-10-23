from pyEnergy.fool import initialize_with_feature_selector

fool = initialize_with_feature_selector("data/ChangErZhai-40-139079-values 20180101-20181031.csv", method="pca")
print(*fool.features())