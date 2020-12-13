# CS-RR

To preprocess MSCOCO training data, from root of repository run:
```
mkdir -p datasets
curl http://images.cocodataset.org/zips/test2017.zip --output MSCOCO.zip
unzip MSCOCO.zip -d datasets
mv datasets/test2017 datasets/MSCOCO
rm MSCOCO.zip
```