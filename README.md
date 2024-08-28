# Classification-using-Neural-Network-with-missing-data-GRU-D

This is a reimplementation of https://github.com/WenjieDu/PyPOTS/tree/main/pypots/classification/grud

GRU-D, short for Gated Recurrent Unit with Decay, is a variant of the GRU (Gated Recurrent Unit) that is specifically designed to handle time-series data with missing values. It was introduced in the paper "Recurrent Neural Networks for Multivariate Time Series with Missing Values" by Zhiqian Chen, Mohammad Taha Bahadori, Edward Choi, Jimeng Sun.

Handling missing data:- 

1. The GRU-D model introduces a decay mechanism to handle missing data effectively. GRU-D modifies the GRU structure by adding learnable decay rates that adjust the influence of the past observations based on how long ago they were recorded. Two types of decay are used- Hidden state decay(gamma-h)  and Input decay(gamma-x).
2. Masking: GRU-D uses a masking mechanism to indicate whether an input value at a specific time step is missing or observed.
3. Imputation: GRU-D  use simple imputation methods (such as Last Observation Carried Forward (LOCF)) to fill in missing values.
4. delta: Delta values measure the elapsed time since the last observed value for each feature in the time-series data. When data is collected irregularly, the delta helps the model understand how long it's been since the last saple of  data was recorded.


Gamma Calculation:

The decay factor gamma is typically calculated using the delta values as follows:
      1. First, the delta is multiplied by a learned weight matrix W and then added to a bias b.
      2. A ReLU activation function is applied to the result.
      3. Finally, the decay factor gamma is obtained by applying the exponential function to the negative of this value, ensuring that the decay factor lies between 0 and 1.

Mathematically:
      gamma=exp(−ReLU(W⋅δ+b))
      
The result is a value that decays exponentially with time, meaning that the longer the time gap (delta), the smaller the value of gamma, which reduces the influence of older data.


Integration in GRU-D:

      1.The delta is passed through the TemporalDecay layers to compute gamma_h and gamma_x.
      2.gamma_h: Controls the decay of the hidden state over time, which is important for deciding how much of the previous hidden state should influence the current state.
      3.gamma_x: Controls the decay of the input features over time, determining how much influence past inputs should have on the current prediction.
