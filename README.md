# Sampling-Theory Studio

## Overview

Sampling-Theory Studio is a desktop application developed to illustrate the fundamental principles of signal sampling and recovery, emphasizing the importance of the Nyquist rate. It allows users to load, sample, and recover signals interactively while exploring various aspects of digital signal processing.

## Features

### 1. **Sample & Recover**
- Load a mid-length signal (around 1000 points) and visualize it.
- Perform sampling at different frequencies, displayed in either actual or normalized terms.
- Recover the signal using the **Whittaker–Shannon interpolation formula**.
- Display four synchronized graphs:
  - Original signal with sampled points.
  - Reconstructed signal.
  - Difference between the original and reconstructed signals.
  - Frequency domain analysis for aliasing detection.

### 2. **Load & Compose**
- Load signals from a file or create them using an in-app signal mixer/composer.
- Add multiple sinusoidal components with customizable frequencies and magnitudes.
- Remove components to refine the mixed signal.

### 3. **Additive Noise**
- Introduce noise to the loaded signal with a user-controllable Signal-to-Noise Ratio (SNR).
- Visualize the impact of noise on the signal frequency.

### 4. **Real-Time Interaction**
- Sampling and recovery operations update in real-time without requiring manual refreshes.

### 5. **Different Reconstruction Methods**
- Explore multiple reconstruction methods, including Whittaker–Shannon interpolation, Zero order hold and Linear Interpolation.
- Select the preferred method from a dropdown menu to compare reconstruction performance.
- Understand the pros and cons of each method with practical examples.

### 6. **Resizable Interface**
- The application UI adjusts seamlessly to different window sizes without losing layout integrity.

### 7. **Diverse Sampling Scenarios**
- Includes at least three predefined synthetic test signals:
  1. A mix of 2Hz and 6Hz sinusoids, demonstrating recovery at 12Hz and aliasing effects at 4Hz or 8Hz sampling rates.
  2. Additional test cases illustrating tricky or unique scenarios, exploiting potential pitfalls in signal recovery.

## Demo video



https://github.com/user-attachments/assets/b614c1ab-ddba-43b5-ac86-80fe0990167a


### Technologies Used
- Python
- PyQt
- Pyqtgraph
- Required libraries/modules: (list dependencies, e.g., `numpy`, `matplotlib`, `scipy`, etc.)

### Setup Instructions
1. Clone this repository:
   ```bash
   git clone https://github.com/seiftaha/sampling-theory-studio.git

   
## Contributors

- [Saif mohamed](https://github.com/seiftaha)
- [Mazen marwan](https://github.com/Mazenmarwan023)
- [Mahmoud mohamed](https://github.com/mahmouddmo22)
- [Farha](https://github.com/farha1010)
- [Eman emad](https://github.com/alyaaa20)

## License

This project is open-source and available under the [MIT License](LICENSE).
