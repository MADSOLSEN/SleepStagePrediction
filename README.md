![Conceptual visualization of the proposed Deep Learning Framework for Sleep Stage Classification using Accelerometry and Photoplethysmography acquired from Consumer Sleep Technologies](/resources/images/model_ver15.png)
Conceptual representation of the proposed deep neural network (DNN) in an example recording. Two time-aligned spectrograms are firstly concatenated, reshaped, and zero-padded to conform to the subsequent temporal module. Then the segments are processed in the deep convolutional neural network, inspired by U-Net [^2], [^3], [^4] that consists of 𝑀 encoder and decoder blocks. Finally, the output is segmented into sleep epochs
of 30 s duration and classified into 4 classes: wake, light sleep, deep sleep. The classification module is inspired by the segment classifier from U-Sleep [^3]. The argmax of the model predictions is presented along with the ground truth hypnogram for comparison. Periods with data loss are labeled with mask. 𝑀: Number of encoder and decoder blocks in U-net; 𝑇: number of sleep epochs; 𝑁: duration in seconds of the recording; GELU: Gaussian Error Linear Unit activation function; conv: convolution; convTranspose: transposed convolutional; batch norm: batch normalization;
STFT: Short Time Fourier Transform; ACC: Accelerometry; PPG: Photoplethysmography


# A Flexible Deep Learning Architecture for Temporal Sleep Stage Classification using Accelerometry and Photoplethysmography[^1]

[[Paper](https://ieeexplore.ieee.org/document/9813567)|[Presentation](Link)]

[^1]: Olsen, Mads, Jamie M. Zeitzer, Risa N. Richardson, Polina Davidenko, Poul J. Jennum, Helge BD Sørensen, and Emmanuel Mignot. "A flexible deep learning architecture for temporal sleep stage classification using accelerometry and photoplethysmography." IEEE Transactions on Biomedical Engineering (2022).
[^2] M. Perslev, S. Darkner, L. Kempfner, M. Nikolic, P. J. Jennum, and C. Igel, “U-Sleep: resilient high-frequency sleep staging,” npj Digit. Med., vol. 4, no. 1, pp. 1–12, 2021.
[^3] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” Lect. Notes Comput. Sci. (including Subser. Lect. Notes Artif. Intell. Lect. Notes Bioinformatics), vol. 9351, pp. 234–241, 2015.
[^4] H. Li and Y. Guan, “DeepSleep convolutional neural network allows accurate and fast detection of sleep arousal,” Commun. Biol., vol. 4, no. 1, pp. 1–11, 2021. 

-------------------------------------------------------------------------------------------------------------------------------------------------------------
## Minimal example
Use minimal_example to get started. 


-------------------------------------------------------------------------------------------------------------------------------------------------------------
## Reproduction of all experiments in paper
Updates will be coming soon. 

