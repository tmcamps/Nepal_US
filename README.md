# Quality Assurance Ultrasound Transducers based on Reverberation Pattern in Air

**Input:** Air reverberation pattern extracted from US transducer, using QA preset. 
* .BMP file
* Saved in /analysis/data/{US device}/{US probe}/
* See /analysis/data directory as example for both devices

**Packages:** the following packages have to be installed to run the analysis
* numpy 
* scipy
* os
* yaml
* pydicom
* PIL
* operator
* matplotlib

**Settings:** the settings per device and probe are given in .yaml files in the /analysis/settings/ directory

**Run analysis:** after installing the required packages and uploading the images correctly to the /analysis/data directory, the analysis is 
executed by running the __main__.py file in Python

**Output:** Overview image saved to /analysis/results/{US device}/{US probe} showing:
* selected pattern
* depth-reverberation pattern including reverberation lines 
* horizontal reverberation pattern including dead and weak elements
* the QA settings that have been used

