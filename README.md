# SteadySeg
This project is the official implementation of SteadySeg: Steadiness Recognition for Maritime Trajectory Segmentation with Cross-Training

The project is conducted based on data provided by Danish Maritime Authority (DMA) (https://dma.dk/safety-at-sea/navigational-information/ais-data) and the Fleet Register, European Commission (https://webgate.ec.europa.eu/fleet-europa/search_en)

### Data
Both the dataset mentioned in the paper and the trained model are provided through this link: https://sites.google.com/view/steadyseg/home

For demo purpose, we only provide a subset of our dataset under ./data. For reproducibility, please download the whole training and testing dataset from the website provided above.

### Training
````
python train.py
````

### Testing
````
python test.py
````

