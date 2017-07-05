# WSS2017_Project

## Project Abstract:

This project predicts physical properties of organic compounds based on chemical structure. In particular, the code estimates melting point, boiling point, heat of fusion, heat of vaporization, heat of combustion, and heat of formation. These properties are determined from chemical structure and properties that can easily be calculated from structure. They do not require any experimental or quantum mechanical data as inputs. 


Mathematica includes a large repository of chemical data that proves useful for this project, with around 44,000 entries. These data were filtered to remove missing entries, as well as those containing elements other than Carbon, Hydrogen, Nitrogen, and Oxygen, leaving around 5500 samples. Most of the work lay in processing the data and extracting useful features. Features included the molecule's geometry, topology, functional groups and substructures, moments of inertia, as well as a host of other features. Once the data was gathered and saved, properties were predicted by the random forest algorithm, which performed better than neural networks. With that said, it is possible that a more carefully designed neural network would be able to predict chemical properties more accurately.


The machine learning algorithm predicted melting point with moderate accuracy. The MAE of the predictions was about 30 K, with an average relative error of 10%. Predictions of other properties fared much the same. It may be that accurate prediction from structure has a limit, and quantum mechanical information is required for accuracy. Future studies should include estimates of dipole moments, as well as a broader range of organic molecules (e.g., compounds containing Sulfur, Phosphorous, and halogens). Further improvements can be made by describing how molecules fit together in 3D space.

## Files:
- The preprocessing file creates a local dataset (~50 MB) that is used for prediction. It can take quite a while to run, so it is recommended that the user process the data incrementally. 
 - The predict file uses this dataset to construct a model for certain chemical properties.
 - MarcThomson_FP contains a general summary of procedure and results.

